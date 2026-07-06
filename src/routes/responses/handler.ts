import type { Context } from "hono"

import { streamSSE } from "hono/streaming"

import {
  isResponsesApiWebSearchEnabled as isConfiguredResponsesApiWebSearchEnabled,
  resolveMappedModel,
} from "~/lib/config"
import { createHandlerLogger, debugJson, debugJsonTail } from "~/lib/logger"
import { parseProviderModelAlias } from "~/lib/provider-model"
import { handleProviderResponsesForProvider } from "~/routes/provider/responses/handler"
import { state } from "~/lib/state"
import {
  createCopilotTokenUsageRecorder,
  normalizeAnthropicUsage,
  normalizeOpenAIUsage,
  normalizeOptionalToken,
  normalizeResponsesUsage,
  type UsageTokens,
} from "~/lib/token-usage"
import { generateRequestIdFromPayload, getUUID } from "~/lib/utils"
import type { SubagentMarker } from "~/lib/subagent"
import {
  createResponses as createCopilotResponses,
  type ResponsesPayload,
  type ResponsesResult,
  type ResponseStreamEvent,
} from "~/services/copilot/create-responses"
import { createMessages } from "~/services/copilot/create-messages"
import {
  createChatCompletions,
  type ChatCompletionChunk,
} from "~/services/copilot/create-chat-completions"
import type { AnthropicStreamEventData } from "~/routes/messages/anthropic-types"
import { findEndpointModel } from "~/lib/models"

import { createStreamIdTracker, fixStreamIds } from "./stream-id-sync"
import {
  applyResponsesApiContextManagement,
  compactInputByLatestCompaction,
  getResponsesTransportForModel,
  getResponsesRequestOptions,
  resolveTextVerbosity,
  sanitizeOversizedInputImages,
} from "./utils"
import {
  translateResponsesToMessagesPayload,
  translateAnthropicResultToResponses,
} from "./responses-messages-translation"
import {
  createMessagesToResponsesStreamState,
  translateAnthropicStreamToResponsesEvent,
} from "./responses-messages-stream-translation"
import {
  translateResponsesToCompletionsPayload,
  translateCompletionsResultToResponses,
} from "./responses-completions-translation"
import {
  createResponsesFromCompletionsStreamState,
  translateCompletionsChunkToResponsesEvents,
} from "./responses-completions-stream-translation"
import consola from "consola"

const logger = createHandlerLogger("responses-handler")

export const responsesHandlerDependencies = {
  createResponses: createCopilotResponses,
  isResponsesApiWebSearchEnabled: isConfiguredResponsesApiWebSearchEnabled,
}

export const handleResponses = async (c: Context) => {
  const payload = await c.req.json<ResponsesPayload>()
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
    return await handleProviderResponsesForProvider(c, {
      payload,
      provider: providerModelAlias.provider,
    })
  }

  debugJson(logger, "Responses request payload:", payload)

  const subagentMarker = getCodexResponsesSubagentMarker(c)
  if (subagentMarker) {
    debugJson(logger, "Detected Codex subagent headers:", subagentMarker)
  }

  const incomingSessionId = getIncomingResponsesSessionId(c)
  const sessionId = incomingSessionId ? getUUID(incomingSessionId) : undefined
  const requestId = generateRequestIdFromPayload(
    { messages: payload.input },
    sessionId,
  )
  logger.debug("Generated request ID:", requestId)

  const fallbackSessionId = sessionId ?? getUUID(requestId)
  logger.debug("Extracted session ID:", fallbackSessionId)
  const recordUsage = createCopilotTokenUsageRecorder({
    endpoint: "responses",
    fallbackSessionId,
    model: payload.model,
  })

  removeUnsupportedTools(payload)

  if (!responsesHandlerDependencies.isResponsesApiWebSearchEnabled()) {
    removeWebSearchTool(payload)
  }

  compactInputByLatestCompaction(payload)

  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )
  const responsesTransport = getResponsesTransportForModel(selectedModel)

  if (!responsesTransport) {
    // Model doesn't support /responses - check if it supports /v1/messages
    const selectedModelObj = findEndpointModel(payload.model)
    const supportsMessages =
      selectedModelObj?.supported_endpoints?.includes("/v1/messages") ?? false

    if (supportsMessages) {
      return await handleWithMessagesApi(c, payload)
    }

    // Check if it supports /chat/completions
    const supportsChatCompletions =
      selectedModelObj?.supported_endpoints?.includes("/chat/completions")
      ?? false
    if (supportsChatCompletions) {
      return await handleWithCompletionsApi(c, payload)
    }

    return c.json(
      {
        error: {
          message:
            "This model does not support the responses endpoint. Please choose a different model.",
          type: "invalid_request_error",
        },
      },
      400,
    )
  }

  const sanitizedImageCount = sanitizeOversizedInputImages(
    payload,
    selectedModel?.capabilities.limits.vision?.max_prompt_image_size,
  )
  if (sanitizedImageCount > 0) {
    logger.warn(
      `Omitted ${sanitizedImageCount} oversized input image(s) before forwarding to Copilot Responses`,
    )
  }

  // Smaller than the client compaction threshold, use server-side compaction to maintain cache hit rate
  const maxPromptTokens = selectedModel?.capabilities.limits.max_prompt_tokens
  applyResponsesApiContextManagement(payload, maxPromptTokens, 0.8)

  // Inject text verbosity from config when not already set in the request
  resolveTextVerbosity(payload)

  debugJson(logger, "Translated Responses payload:", payload)

  const { vision, initiator: inferredInitiator } =
    getResponsesRequestOptions(payload)
  const initiator = subagentMarker ? "agent" : inferredInitiator

  const response = await responsesHandlerDependencies.createResponses(payload, {
    vision,
    initiator,
    subagentMarker,
    requestId,
    sessionId: fallbackSessionId,
    transport: responsesTransport,
  })

  if (isStreamingRequested(payload) && isAsyncIterable(response)) {
    logger.debug("Forwarding native Responses stream")
    return streamSSE(c, async (stream) => {
      const idTracker = createStreamIdTracker()
      let usage: UsageTokens = {}

      for await (const chunk of response) {
        debugJson(logger, "Responses stream chunk:", chunk)
        const parsedEvent = parseResponsesStreamEvent(chunk)
        if (
          parsedEvent?.type === "response.completed"
          || parsedEvent?.type === "response.failed"
          || parsedEvent?.type === "response.incomplete"
        ) {
          usage = {
            ...normalizeResponsesUsage(parsedEvent.response.usage),
            total_nano_aiu: normalizeOptionalToken(
              parsedEvent.copilot_usage?.total_nano_aiu,
            ),
          }
        }

        const processedData = fixStreamIds(
          (chunk as { data?: string }).data ?? "",
          (chunk as { event?: string }).event,
          idTracker,
        )

        await stream.writeSSE({
          id: (chunk as { id?: string }).id,
          event: (chunk as { event?: string }).event,
          data: processedData,
        })
      }

      recordUsage(usage)
    })
  }

  debugJsonTail(logger, "Forwarding native Responses result:", {
    value: response,
    tailLength: 400,
  })
  const result = response as ResponsesResult
  recordUsage({
    ...normalizeResponsesUsage(result.usage),
    total_nano_aiu: normalizeOptionalToken(
      result.copilot_usage?.total_nano_aiu,
    ),
  })
  return c.json(result)
}

const isAsyncIterable = <T>(value: unknown): value is AsyncIterable<T> =>
  Boolean(value)
  && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === "function"

const isStreamingRequested = (payload: ResponsesPayload): boolean =>
  Boolean(payload.stream)

const parseResponsesStreamEvent = (
  chunk: unknown,
): ResponseStreamEvent | null => {
  const data = (chunk as { data?: string }).data
  if (!data || data === "[DONE]") {
    return null
  }

  try {
    return JSON.parse(data) as ResponseStreamEvent
  } catch {
    return null
  }
}

const removeWebSearchTool = (payload: ResponsesPayload): void => {
  if (!Array.isArray(payload.tools) || payload.tools.length === 0) return

  payload.tools = payload.tools.filter((t) => {
    return t.type !== "web_search"
  })
}

const COPILOT_UNSUPPORTED_TOOL_TYPES = new Set(["image_generation"])

export const removeUnsupportedTools = (payload: ResponsesPayload): void => {
  if (!Array.isArray(payload.tools) || payload.tools.length === 0) return

  const dropped: Array<string> = []
  payload.tools = payload.tools.filter((t) => {
    const type = t.type as string
    if (COPILOT_UNSUPPORTED_TOOL_TYPES.has(type)) {
      dropped.push(type)
      return false
    }
    return true
  })
  if (dropped.length > 0) {
    logger.debug("Removed unsupported tools:", dropped)
  }
}

const getIncomingResponsesSessionId = (c: Context): string | undefined =>
  getTrimmedHeader(c, "session-id") ?? getTrimmedHeader(c, "x-session-id")

const codexSubagentHeaderValues = new Set([
  "collab_spawn",
  "compact",
  "memory_consolidation",
  "review",
])

const getCodexResponsesSubagentMarker = (c: Context): SubagentMarker | null => {
  const agentType = getTrimmedHeader(c, "x-openai-subagent")
  if (!agentType || !codexSubagentHeaderValues.has(agentType)) {
    return null
  }

  const threadId = getTrimmedHeader(c, "thread-id")
  const rootSessionId = getIncomingResponsesSessionId(c)
  const parentThreadId = getTrimmedHeader(c, "x-codex-parent-thread-id")
  if (!threadId && !rootSessionId && !parentThreadId) {
    return null
  }

  const agentId = threadId ?? parentThreadId ?? rootSessionId ?? agentType

  return {
    agent_id: agentId,
    agent_type: agentType,
    session_id: threadId ?? rootSessionId ?? agentId,
  }
}

const getTrimmedHeader = (c: Context, name: string): string | undefined => {
  const value = c.req.header(name)?.trim()
  return value || undefined
}

// ---------------------------------------------------------------------------
// Messages API fallback for models that support /v1/messages but not /responses
// ---------------------------------------------------------------------------

const handleWithMessagesApi = async (c: Context, payload: ResponsesPayload) => {
  const messagesPayload = translateResponsesToMessagesPayload(payload)

  // Detect if we injected a synthetic tool for structured output
  const textFormat = (payload.text as { format?: { type: string } } | undefined)
    ?.format
  const forcedToolName =
    textFormat?.type === "json_schema" ? "structured_response" : undefined

  const requestId = generateRequestIdFromPayload(
    { messages: payload.input },
    undefined,
  )
  logger.debug("Generated request ID (messages fallback):", requestId)

  const sessionId = getUUID(requestId)
  logger.debug("Extracted session ID (messages fallback):", sessionId)

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
    const responsesResult = translateAnthropicResultToResponses(
      anthropicResult,
      payload.model,
      forcedToolName,
    )
    recordUsage({
      ...normalizeAnthropicUsage(anthropicResult.usage),
      total_nano_aiu: normalizeOptionalToken(
        anthropicResult.copilot_usage?.total_nano_aiu,
      ),
    })
    debugJson(logger, "Translated Responses result:", responsesResult)
    return c.json(responsesResult)
  }

  // Streaming
  logger.debug("Streaming response from Copilot (Messages API fallback)")
  return streamSSE(c, async (stream) => {
    const streamState = createMessagesToResponsesStreamState(forcedToolName)
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

      // Track usage from message_start and message_delta
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

      const responsesEvents = translateAnthropicStreamToResponsesEvent(
        parsedEvent,
        streamState,
      )

      for (const responseEvent of responsesEvents) {
        await stream.writeSSE({
          event: responseEvent.type as string,
          data: JSON.stringify(responseEvent),
        })
      }
    }

    recordUsage(usage)
  })
}

// ---------------------------------------------------------------------------
// Chat Completions API fallback for models that support /chat/completions
// but not /responses or /v1/messages
// ---------------------------------------------------------------------------

const handleWithCompletionsApi = async (
  c: Context,
  payload: ResponsesPayload,
) => {
  const completionsPayload = translateResponsesToCompletionsPayload(payload)

  const requestId = generateRequestIdFromPayload(
    { messages: payload.input },
    undefined,
  )
  logger.debug("Generated request ID (completions fallback):", requestId)

  const sessionId = getUUID(requestId)
  logger.debug("Extracted session ID (completions fallback):", sessionId)

  const recordUsage = createCopilotTokenUsageRecorder({
    endpoint: "chat_completions",
    fallbackSessionId: sessionId,
    model: payload.model,
  })

  debugJson(logger, "Translated Completions payload:", completionsPayload)

  const response = await createChatCompletions(completionsPayload, {
    requestId,
    sessionId,
  })

  // Non-streaming
  if (!payload.stream && !isAsyncIterable(response)) {
    const completionResult = response
    debugJson(logger, "Non-streaming Completions result:", completionResult)
    const responsesResult =
      translateCompletionsResultToResponses(completionResult)
    recordUsage({
      ...normalizeOpenAIUsage(completionResult.usage),
      total_nano_aiu: normalizeOptionalToken(
        completionResult.copilot_usage?.total_nano_aiu,
      ),
    })
    debugJson(logger, "Translated Responses result:", responsesResult)
    return c.json(responsesResult)
  }

  // Streaming
  logger.debug("Streaming response from Copilot (Completions API fallback)")
  return streamSSE(c, async (stream) => {
    const streamState = createResponsesFromCompletionsStreamState()
    let usage: UsageTokens = {}

    for await (const chunk of response as AsyncIterable<{
      event?: string
      data?: string
    }>) {
      const data = chunk.data
      if (!data || data === "[DONE]") {
        continue
      }

      let parsedChunk: ChatCompletionChunk | null = null
      try {
        parsedChunk = JSON.parse(data) as ChatCompletionChunk
      } catch {
        continue
      }

      if (!parsedChunk) continue

      if (parsedChunk.usage || parsedChunk.copilot_usage) {
        usage = {
          ...normalizeOpenAIUsage(parsedChunk.usage),
          total_nano_aiu: normalizeOptionalToken(
            parsedChunk.copilot_usage?.total_nano_aiu,
          ),
        }
      }

      const responsesEvents = translateCompletionsChunkToResponsesEvents(
        parsedChunk,
        streamState,
      )

      for (const responseEvent of responsesEvents) {
        await stream.writeSSE({
          event: responseEvent.type as string,
          data: JSON.stringify(responseEvent),
        })
      }
    }

    recordUsage(usage)
  })
}
