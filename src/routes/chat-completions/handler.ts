import consola from "consola"
import type { Context } from "hono"

import { streamSSE, type SSEMessage } from "hono/streaming"

import { resolveMappedModel } from "~/lib/config"
import { createHandlerLogger, debugJson, debugLazy } from "~/lib/logger"
import { parseProviderModelAlias } from "~/lib/provider-model"
import { state } from "~/lib/state"
import {
  createCopilotTokenUsageRecorder,
  normalizeAnthropicUsage,
  normalizeOpenAIUsage,
  normalizeOptionalToken,
  normalizeResponsesUsage,
  type UsageTokens,
} from "~/lib/token-usage"
import { generateRequestIdFromPayload, getUUID, isNullish } from "~/lib/utils"
import { handleProviderChatCompletionsForProvider } from "~/routes/provider/chat-completions/handler"
import {
  createChatCompletions,
  type ChatCompletionChunk,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
} from "~/services/copilot/create-chat-completions"
import {
  createResponses,
  type ResponseStreamEvent,
} from "~/services/copilot/create-responses"
import { createMessages } from "~/services/copilot/create-messages"
import type { AnthropicStreamEventData } from "~/routes/messages/anthropic-types"
import {
  getResponsesTransportForModel,
  getResponsesRequestOptions,
} from "~/routes/responses/utils"
import {
  translateCompletionsToResponsesPayload,
  translateResponsesResultToCompletions,
} from "./completions-responses-translation"
import {
  createCompletionsStreamState,
  translateResponsesStreamEventToCompletions,
} from "./completions-responses-stream-translation"
import {
  translateCompletionsToMessagesPayload,
  translateAnthropicResultToCompletions,
} from "./completions-messages-translation"
import {
  createCompletionsFromMessagesStreamState,
  translateAnthropicStreamToCompletions,
} from "./completions-messages-stream-translation"

const logger = createHandlerLogger("chat-completions-handler")

export async function handleCompletion(c: Context) {
  let payload = await c.req.json<ChatCompletionsPayload>()
  const requestedModel = payload.model
  payload.model = resolveMappedModel(payload.model)
  if (payload.model !== requestedModel) {
    consola.debug(
      `Resolved model mapping: ${requestedModel} -> ${payload.model}`,
    )
  }

  const providerModelAlias = parseProviderModelAlias(payload.model)
  if (providerModelAlias) {
    payload.model = providerModelAlias.model
    return await handleProviderChatCompletionsForProvider(c, {
      payload,
      provider: providerModelAlias.provider,
    })
  }

  debugJson(logger, "Request payload:", payload)

  // Find the selected model
  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )

  if (
    isNullish(payload.max_tokens)
    && isNullish(payload.max_completion_tokens)
  ) {
    payload = {
      ...payload,
      max_tokens: selectedModel?.capabilities.limits.max_output_tokens,
    }
    debugJson(logger, "Set max_tokens to:", payload.max_tokens)
  }

  if (payload.model.includes("gpt")) {
    if (isNullish(payload.max_completion_tokens)) {
      payload.max_completion_tokens = payload.max_tokens
    }
    delete payload.max_tokens
  }

  // Check if this model requires the Responses API (supports /responses but NOT /chat/completions)
  const responsesTransport = getResponsesTransportForModel(selectedModel)
  const supportsChatCompletions =
    selectedModel?.supported_endpoints?.includes("/chat/completions") ?? false
  if (responsesTransport && !supportsChatCompletions) {
    return await handleWithResponsesApi(c, payload)
  }

  // Check if this model only supports Messages API (supports /v1/messages but NOT /chat/completions)
  const supportsMessages =
    selectedModel?.supported_endpoints?.includes("/v1/messages") ?? false
  if (!supportsChatCompletions && supportsMessages) {
    return await handleWithMessagesApi(c, payload)
  }

  // not support subagent marker for now , set sessionId = getUUID(requestId)
  const requestId = generateRequestIdFromPayload(payload)
  logger.debug("Generated request ID:", requestId)

  const sessionId = getUUID(requestId)
  logger.debug("Extracted session ID:", sessionId)
  const recordUsage = createCopilotTokenUsageRecorder({
    endpoint: "chat_completions",
    fallbackSessionId: sessionId,
    model: payload.model,
  })

  const response = await createChatCompletions(payload, {
    requestId,
    sessionId,
  })

  if (isNonStreaming(response)) {
    debugJson(logger, "Non-streaming response:", response)
    recordUsage({
      ...normalizeOpenAIUsage(response.usage),
      total_nano_aiu: normalizeOptionalToken(
        response.copilot_usage?.total_nano_aiu,
      ),
    })
    return c.json(response)
  }

  logger.debug("Streaming response")
  return streamSSE(c, async (stream) => {
    let usage: UsageTokens = {}

    for await (const chunk of response) {
      debugJson(logger, "Streaming chunk:", chunk)
      const parsedChunk = parseChatCompletionChunk(chunk)
      if (parsedChunk?.usage || parsedChunk?.copilot_usage) {
        usage = {
          ...normalizeOpenAIUsage(parsedChunk.usage),
          total_nano_aiu: normalizeOptionalToken(
            parsedChunk.copilot_usage?.total_nano_aiu,
          ),
        }
      }
      await stream.writeSSE(chunk as SSEMessage)
    }

    recordUsage(usage)
  })
}

// ---------------------------------------------------------------------------
// Responses API flow for models that don't support /chat/completions
// ---------------------------------------------------------------------------

const handleWithResponsesApi = async (
  c: Context,
  payload: ChatCompletionsPayload,
) => {
  const responsesPayload = translateCompletionsToResponsesPayload(payload)

  const requestId = generateRequestIdFromPayload(payload)
  logger.debug("Generated request ID (responses flow):", requestId)

  const sessionId = getUUID(requestId)
  logger.debug("Extracted session ID (responses flow):", sessionId)

  const recordUsage = createCopilotTokenUsageRecorder({
    endpoint: "responses",
    fallbackSessionId: sessionId,
    model: payload.model,
  })

  debugJson(logger, "Translated Responses payload:", responsesPayload)

  const { vision, initiator } = getResponsesRequestOptions(responsesPayload)
  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )
  const transport = getResponsesTransportForModel(selectedModel) ?? "http"

  const response = await createResponses(responsesPayload, {
    vision,
    initiator,
    requestId,
    sessionId,
    transport,
  })

  // Non-streaming
  if (!payload.stream && !isAsyncIterable(response)) {
    const result = response
    debugJson(logger, "Non-streaming Responses result:", result)
    const completionResponse = translateResponsesResultToCompletions(result)
    recordUsage({
      ...normalizeResponsesUsage(result.usage),
      total_nano_aiu: normalizeOptionalToken(
        result.copilot_usage?.total_nano_aiu,
      ),
    })
    debugJson(logger, "Translated Completions response:", completionResponse)
    return c.json(completionResponse)
  }

  // Streaming
  logger.debug("Streaming response from Copilot (Responses API)")
  return streamSSE(c, async (stream) => {
    const streamState = createCompletionsStreamState()
    let usage: UsageTokens = {}

    for await (const chunk of response as AsyncIterable<{
      data?: string
      event?: string
    }>) {
      const eventName = chunk.event
      if (eventName === "ping") {
        continue
      }

      const data = chunk.data
      if (!data) {
        continue
      }

      debugLazy(logger, () => ["Responses raw stream event:", data])

      const responseEvent = JSON.parse(data) as ResponseStreamEvent
      if (
        responseEvent.type === "response.completed"
        || responseEvent.type === "response.failed"
        || responseEvent.type === "response.incomplete"
      ) {
        usage = {
          ...normalizeResponsesUsage(responseEvent.response.usage),
          total_nano_aiu: normalizeOptionalToken(
            responseEvent.copilot_usage?.total_nano_aiu,
          ),
        }
      }

      const sseMessages = translateResponsesStreamEventToCompletions(
        responseEvent,
        streamState,
      )
      for (const msg of sseMessages) {
        if (msg.data) {
          debugLazy(logger, () => ["Translated Completions chunk:", msg.data])
          await stream.writeSSE(msg as SSEMessage)
        }
      }
    }

    recordUsage(usage)
  })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")

const isAsyncIterable = <T>(value: unknown): value is AsyncIterable<T> =>
  Boolean(value)
  && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === "function"

const parseChatCompletionChunk = (
  chunk: unknown,
): ChatCompletionChunk | null => {
  const data = (chunk as { data?: string }).data
  if (!data || data === "[DONE]") {
    return null
  }

  try {
    return JSON.parse(data) as ChatCompletionChunk
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// Messages API flow for models that only support /v1/messages
// ---------------------------------------------------------------------------

const handleWithMessagesApi = async (
  c: Context,
  payload: ChatCompletionsPayload,
) => {
  const messagesPayload = translateCompletionsToMessagesPayload(payload)

  // Detect if we injected a synthetic tool for structured output
  const rf = payload.response_format
  const forcedToolName =
    rf && rf.type === "json_schema" ? "structured_response" : undefined

  const requestId = generateRequestIdFromPayload(payload)
  logger.debug("Generated request ID (messages flow):", requestId)

  const sessionId = getUUID(requestId)
  logger.debug("Extracted session ID (messages flow):", sessionId)

  const recordUsage = createCopilotTokenUsageRecorder({
    endpoint: "messages",
    fallbackSessionId: sessionId,
    model: payload.model,
  })

  debugJson(logger, "Translated Messages payload:", messagesPayload)

  const response = await createMessages(messagesPayload, undefined, {
    requestId,
    sessionId,
  })

  // Non-streaming
  if (!payload.stream && !isAsyncIterable(response)) {
    const anthropicResult = response
    debugJson(logger, "Non-streaming Messages result:", anthropicResult)
    const completionResponse = translateAnthropicResultToCompletions(
      anthropicResult,
      forcedToolName,
    )
    recordUsage({
      ...normalizeAnthropicUsage(anthropicResult.usage),
      total_nano_aiu: normalizeOptionalToken(
        anthropicResult.copilot_usage?.total_nano_aiu,
      ),
    })
    debugJson(logger, "Translated Completions response:", completionResponse)
    return c.json(completionResponse)
  }

  // Streaming
  logger.debug("Streaming response from Copilot (Messages API)")
  return streamSSE(c, async (stream) => {
    const streamState = createCompletionsFromMessagesStreamState(forcedToolName)
    let usage: UsageTokens = {}

    for await (const chunk of response as AsyncIterable<{
      event?: string
      data?: string
    }>) {
      const data = chunk.data
      if (!data || data === "[DONE]") {
        continue
      }

      let parsedEvent: AnthropicStreamEventData | null = null
      try {
        parsedEvent = JSON.parse(data) as AnthropicStreamEventData
      } catch {
        continue
      }

      if (!parsedEvent) continue

      // Track usage
      if (parsedEvent.type === "message_start") {
        usage = {
          ...normalizeAnthropicUsage(parsedEvent.message.usage),
          total_nano_aiu: normalizeOptionalToken(
            (
              parsedEvent.message as {
                copilot_usage?: { total_nano_aiu?: number }
              }
            ).copilot_usage?.total_nano_aiu,
          ),
        }
      } else if (parsedEvent.type === "message_delta") {
        if (parsedEvent.usage) {
          usage = {
            ...usage,
            ...normalizeAnthropicUsage(parsedEvent.usage),
            total_nano_aiu: normalizeOptionalToken(
              parsedEvent.copilot_usage?.total_nano_aiu,
            ),
          }
        }
      }

      const sseMessages = translateAnthropicStreamToCompletions(
        parsedEvent,
        streamState,
      )
      for (const msg of sseMessages) {
        if (msg.data) {
          await stream.writeSSE(msg as SSEMessage)
        }
      }
    }

    recordUsage(usage)
  })
}
