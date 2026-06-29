/**
 * Translate OpenAI Chat Completions payloads to Anthropic Messages API payloads
 * and translate Anthropic responses back to Chat Completions format.
 *
 * This provides a direct completions -> messages path without going through
 * the Responses API as an intermediary.
 *
 * Structured output: when the caller requests `response_format.json_schema`,
 * the schema is enforced via a synthetic forced tool_use tool on the Anthropic
 * side (mirroring Hindsight's AnthropicLLM strict_schema pattern). The tool's
 * validated input is unwrapped and returned as text content to the caller.
 */

import type {
  AnthropicAssistantContentBlock,
  AnthropicImageBlock,
  AnthropicMessage,
  AnthropicMessagesPayload,
  AnthropicResponse,
  AnthropicTool,
  AnthropicUserContentBlock,
} from "~/routes/messages/anthropic-types"

import type {
  ChatCompletionResponse,
  ChatCompletionsPayload,
  ContentPart,
  Message,
  Tool,
} from "~/services/copilot/create-chat-completions"

// Name of the synthetic tool used for structured output enforcement
const STRUCTURED_TOOL_NAME = "structured_response"

// ---------------------------------------------------------------------------
// Structured output detection
// ---------------------------------------------------------------------------

interface JsonSchemaSpec {
  name: string
  schema: Record<string, unknown>
  strict?: boolean
}

const extractJsonSchema = (
  payload: ChatCompletionsPayload,
): JsonSchemaSpec | null => {
  const rf = payload.response_format
  if (!rf || rf.type !== "json_schema") return null
  const spec = rf as {
    type: "json_schema"
    json_schema: {
      name: string
      schema: Record<string, unknown>
      strict?: boolean
    }
  }
  return spec.json_schema
}

// ---------------------------------------------------------------------------
// Payload: Completions -> Messages
// ---------------------------------------------------------------------------

export const translateCompletionsToMessagesPayload = (
  payload: ChatCompletionsPayload,
): AnthropicMessagesPayload => {
  const { system, messages } = translateMessages(payload.messages)
  const jsonSchema = extractJsonSchema(payload)
  const tools = translateTools(payload.tools)
  const toolChoice = translateToolChoice(payload.tool_choice)

  const messagesPayload: AnthropicMessagesPayload = {
    model: payload.model,
    messages,
    max_tokens: payload.max_completion_tokens ?? payload.max_tokens ?? 8192,
    stream: payload.stream ?? undefined,
    ...(system ? { system } : {}),
    ...(tools && tools.length > 0 ? { tools } : {}),
    ...(toolChoice ? { tool_choice: toolChoice } : {}),
    ...(payload.top_p != null ? { top_p: payload.top_p } : {}),
    ...(payload.user ? { metadata: { user_id: payload.user } } : {}),
  }

  // Inject synthetic forced tool for structured output
  if (jsonSchema) {
    const syntheticTool: AnthropicTool = {
      name: STRUCTURED_TOOL_NAME,
      description: "Return the structured response.",
      input_schema: jsonSchema.schema,
    }
    messagesPayload.tools = [...(messagesPayload.tools ?? []), syntheticTool]
    messagesPayload.tool_choice = {
      type: "tool",
      name: STRUCTURED_TOOL_NAME,
    }
  }

  return messagesPayload
}

// ---------------------------------------------------------------------------
// Result: Anthropic response -> Completions
// ---------------------------------------------------------------------------

export const translateAnthropicResultToCompletions = (
  response: AnthropicResponse,
  forcedToolName?: string,
): ChatCompletionResponse => {
  let content = ""
  const toolCalls: Array<{
    id: string
    type: "function"
    function: { name: string; arguments: string }
  }> = []

  for (const block of response.content) {
    if (block.type === "text") {
      content += block.text
    } else if (block.type === "tool_use") {
      const toolUse = block
      if (forcedToolName && toolUse.name === forcedToolName) {
        // Synthetic tool - unwrap input as text content
        content += JSON.stringify(toolUse.input)
      } else {
        toolCalls.push({
          id: toolUse.id,
          type: "function",
          function: {
            name: toolUse.name,
            arguments: JSON.stringify(toolUse.input),
          },
        })
      }
    }
    // thinking blocks are not part of the completions response
  }

  // If we unwrapped a synthetic tool and there are no real tool calls,
  // the finish reason should be "stop", not "tool_calls"
  const suppressToolUseStopReason =
    forcedToolName !== undefined && toolCalls.length === 0
  const finishReason =
    suppressToolUseStopReason ? "stop" : (
      mapStopReason(response.stop_reason, toolCalls.length > 0)
    )

  return {
    id: response.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: content || null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        logprobs: null,
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens:
        (response.usage.input_tokens ?? 0)
        + (response.usage.cache_read_input_tokens ?? 0),
      completion_tokens: response.usage.output_tokens ?? 0,
      total_tokens:
        (response.usage.input_tokens ?? 0)
        + (response.usage.cache_read_input_tokens ?? 0)
        + (response.usage.output_tokens ?? 0),
      ...(response.usage.cache_read_input_tokens != null ?
        {
          prompt_tokens_details: {
            cached_tokens: response.usage.cache_read_input_tokens,
          },
        }
      : {}),
    },
    ...(response.copilot_usage ?
      { copilot_usage: response.copilot_usage }
    : {}),
  }
}

// ---------------------------------------------------------------------------
// Messages translation
// ---------------------------------------------------------------------------

const translateMessages = (
  messages: Array<Message>,
): { system: string | null; messages: Array<AnthropicMessage> } => {
  let system: string | null = null
  const result: Array<AnthropicMessage> = []

  let pendingUserContent: Array<AnthropicUserContentBlock> = []
  let pendingAssistantContent: Array<AnthropicAssistantContentBlock> = []

  const flushUser = () => {
    if (pendingUserContent.length > 0) {
      result.push({ role: "user", content: [...pendingUserContent] })
      pendingUserContent = []
    }
  }

  const flushAssistant = () => {
    if (pendingAssistantContent.length > 0) {
      result.push({ role: "assistant", content: [...pendingAssistantContent] })
      pendingAssistantContent = []
    }
  }

  for (const message of messages) {
    if (message.role === "system" || message.role === "developer") {
      const text = extractTextContent(message.content)
      if (text) {
        system = system ? `${system}\n\n${text}` : text
      }
      continue
    }

    if (message.role === "tool") {
      // Tool results go into user content
      flushAssistant()
      pendingUserContent.push({
        type: "tool_result",
        tool_use_id: message.tool_call_id ?? "",
        content: stringifyContent(message.content),
      })
      continue
    }

    if (message.role === "user") {
      flushAssistant()
      const blocks = translateUserContent(message.content)
      pendingUserContent.push(...blocks)
      continue
    }

    if (message.role === "assistant") {
      flushUser()
      // Add text content
      const text = extractTextContent(message.content)
      if (text) {
        pendingAssistantContent.push({ type: "text", text })
      }
      // Add tool calls
      if (message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          pendingAssistantContent.push({
            type: "tool_use",
            id: toolCall.id,
            name: toolCall.function.name,
            input: parseArguments(toolCall.function.arguments),
          })
        }
      }
      continue
    }
  }

  flushUser()
  flushAssistant()

  // Ensure we have at least one message
  if (result.length === 0) {
    result.push({ role: "user", content: "Hello" })
  }

  // Ensure alternating roles
  return { system, messages: ensureAlternating(result) }
}

// ---------------------------------------------------------------------------
// Content helpers
// ---------------------------------------------------------------------------

const extractTextContent = (content: Message["content"]): string | null => {
  if (typeof content === "string") {
    return content
  }
  if (!content || content.length === 0) {
    return null
  }
  const text = content
    .filter(
      (part): part is Extract<ContentPart, { type: "text" }> =>
        part.type === "text",
    )
    .map((part) => part.text)
    .join("\n\n")
  return text || null
}

const translateUserContent = (
  content: Message["content"],
): Array<AnthropicUserContentBlock> => {
  if (typeof content === "string") {
    return content.length > 0 ? [{ type: "text", text: content }] : []
  }
  if (!content || content.length === 0) {
    return []
  }

  const blocks: Array<AnthropicUserContentBlock> = []
  for (const part of content) {
    if (part.type === "text") {
      blocks.push({ type: "text", text: part.text })
    } else if (part.type === "image_url") {
      const parsed = parseDataUrl(part.image_url.url)
      if (parsed) {
        blocks.push({
          type: "image",
          source: {
            type: "base64",
            media_type:
              parsed.mediaType as AnthropicImageBlock["source"]["media_type"],
            data: parsed.data,
          },
        })
      }
    }
    // file parts: Anthropic supports document blocks but only for PDF
    // Skip for now - most completions clients don't send file parts
  }

  return blocks.length > 0 ? blocks : [{ type: "text", text: "" }]
}

const stringifyContent = (content: Message["content"]): string => {
  if (typeof content === "string") {
    return content
  }
  if (!content) {
    return ""
  }
  const text = content
    .filter(
      (part): part is Extract<ContentPart, { type: "text" }> =>
        part.type === "text",
    )
    .map((part) => part.text)
    .join("\n\n")
  return text || JSON.stringify(content)
}

const parseArguments = (args: string): Record<string, unknown> => {
  if (!args || args.trim().length === 0) return {}
  try {
    const parsed: unknown = JSON.parse(args)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return { arguments: parsed }
  } catch {
    return {}
  }
}

const parseDataUrl = (
  dataUrl: string,
): { mediaType: string; data: string } | null => {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/)
  if (!match) return null
  return { mediaType: match[1], data: match[2] }
}

// ---------------------------------------------------------------------------
// Tools translation
// ---------------------------------------------------------------------------

const translateTools = (
  tools: Array<Tool> | null | undefined,
): Array<AnthropicTool> | undefined => {
  if (!tools || tools.length === 0) {
    return undefined
  }
  return tools.map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters ?? {
      type: "object",
      properties: {},
    },
  }))
}

const translateToolChoice = (
  toolChoice: ChatCompletionsPayload["tool_choice"],
): AnthropicMessagesPayload["tool_choice"] | undefined => {
  if (!toolChoice) return undefined

  if (typeof toolChoice === "string") {
    switch (toolChoice) {
      case "auto":
        return { type: "auto" }
      case "required":
        return { type: "any" }
      case "none":
        return { type: "none" }
      default:
        return { type: "auto" }
    }
  }

  // { type: "function", function: { name: "..." } }
  if (toolChoice.function?.name) {
    return { type: "tool", name: toolChoice.function.name }
  }

  return { type: "auto" }
}

// ---------------------------------------------------------------------------
// Stop reason mapping
// ---------------------------------------------------------------------------

const mapStopReason = (
  stopReason: AnthropicResponse["stop_reason"],
  hasToolCalls: boolean,
): "stop" | "length" | "tool_calls" | "content_filter" => {
  if (hasToolCalls || stopReason === "tool_use") {
    return "tool_calls"
  }
  if (stopReason === "max_tokens") {
    return "length"
  }
  return "stop"
}

// ---------------------------------------------------------------------------
// Alternating messages enforcement
// ---------------------------------------------------------------------------

const ensureAlternating = (
  messages: Array<AnthropicMessage>,
): Array<AnthropicMessage> => {
  if (messages.length <= 1) return messages

  const result: Array<AnthropicMessage> = [messages[0]]

  for (let i = 1; i < messages.length; i++) {
    const current = messages[i]
    const previous = result[result.length - 1]

    if (current.role === previous.role) {
      // Merge content into previous message
      const prevContent =
        typeof previous.content === "string" ?
          [{ type: "text" as const, text: previous.content }]
        : (previous.content as Array<AnthropicUserContentBlock>)
      const currContent =
        typeof current.content === "string" ?
          [{ type: "text" as const, text: current.content }]
        : (current.content as Array<AnthropicUserContentBlock>)

      ;(previous as { content: Array<AnthropicUserContentBlock> }).content = [
        ...prevContent,
        ...currContent,
      ]
    } else {
      result.push(current)
    }
  }

  return result
}
