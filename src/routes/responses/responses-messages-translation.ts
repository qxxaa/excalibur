/**
 * Responses API → Anthropic Messages API translation layer.
 *
 * When a /v1/responses request targets a model that only supports /v1/messages,
 * this module translates the Responses payload into an Anthropic Messages payload,
 * sends it upstream, and translates the Anthropic response back to Responses format.
 *
 * This is the reverse direction of messages/responses-translation.ts.
 */

import {
  type AnthropicAssistantContentBlock,
  type AnthropicMessagesPayload,
  type AnthropicResponse,
  type AnthropicTool,
  type AnthropicMessage,
  type AnthropicUserMessage,
  type AnthropicAssistantMessage,
  type AnthropicUserContentBlock,
  type AnthropicImageBlock,
} from "~/routes/messages/anthropic-types"

import {
  type FunctionTool,
  type ResponseFunctionCallOutputItem,
  type ResponseFunctionToolCallItem,
  type ResponseInputContent,
  type ResponseInputImage,
  type ResponseInputFile,
  type ResponseInputMessage,
  type ResponseInputReasoning,
  type ResponseInputText,
  type ResponseOutputFunctionCall,
  type ResponseOutputItem,
  type ResponseOutputMessage,
  type ResponseOutputReasoning,
  type ResponsesPayload,
  type ResponsesResult,
} from "~/services/copilot/create-responses"

// ---------------------------------------------------------------------------
// Request translation: Responses payload → Anthropic Messages payload
// ---------------------------------------------------------------------------

export const translateResponsesToMessagesPayload = (
  payload: ResponsesPayload,
): AnthropicMessagesPayload => {
  const system = payload.instructions ?? undefined
  const messages = translateInputToMessages(payload.input)
  const tools = translateToolsToAnthropic(payload.tools)
  const toolChoice = translateToolChoiceToAnthropic(payload.tool_choice)

  const messagesPayload: AnthropicMessagesPayload = {
    model: payload.model,
    messages,
    max_tokens: payload.max_output_tokens ?? 8192,
    stream: payload.stream ?? undefined,
    ...(system ? { system } : {}),
    ...(tools && tools.length > 0 ? { tools } : {}),
    ...(toolChoice ? { tool_choice: toolChoice } : {}),
    ...(payload.top_p != null ? { top_p: payload.top_p } : {}),
    ...(payload.metadata ? { metadata: payload.metadata } : {}),
  }

  // Map reasoning effort to thinking config
  if (payload.reasoning?.effort && payload.reasoning.effort !== "none") {
    messagesPayload.thinking = {
      type: "enabled",
      budget_tokens: mapReasoningEffortToBudget(
        payload.reasoning.effort,
        messagesPayload.max_tokens,
      ),
    }
  }

  return messagesPayload
}

const mapReasoningEffortToBudget = (
  effort: string,
  maxTokens: number,
): number => {
  // Scale budget relative to max output tokens
  const budgetMultiplier: Record<string, number> = {
    minimal: 0.5,
    low: 1,
    medium: 2,
    high: 4,
    xhigh: 8,
  }
  const multiplier = budgetMultiplier[effort] ?? 2
  return Math.min(Math.max(Math.round(maxTokens * multiplier), 1024), 128000)
}

// ---------------------------------------------------------------------------
// Input items → Anthropic messages
// ---------------------------------------------------------------------------

const translateInputToMessages = (
  input: ResponsesPayload["input"],
): Array<AnthropicMessage> => {
  if (!input || typeof input === "string") {
    if (typeof input === "string" && input.length > 0) {
      return [{ role: "user", content: input }]
    }
    return [{ role: "user", content: "Hello" }]
  }

  const messages: Array<AnthropicMessage> = []
  let pendingUserContent: Array<AnthropicUserContentBlock> = []
  let pendingAssistantContent: Array<AnthropicAssistantContentBlock> = []

  const flushUser = () => {
    if (pendingUserContent.length > 0) {
      messages.push({ role: "user", content: [...pendingUserContent] })
      pendingUserContent = []
    }
  }

  const flushAssistant = () => {
    if (pendingAssistantContent.length > 0) {
      messages.push({
        role: "assistant",
        content: [...pendingAssistantContent],
      })
      pendingAssistantContent = []
    }
  }

  for (const item of input) {
    const itemType = (item as { type?: string }).type

    if (itemType === "message") {
      const msg = item as ResponseInputMessage
      if (msg.role === "user") {
        flushAssistant()
        const blocks = translateResponseContentToUserBlocks(msg.content)
        pendingUserContent.push(...blocks)
      } else if (msg.role === "assistant") {
        flushUser()
        const blocks = translateResponseContentToAssistantBlocks(msg.content)
        pendingAssistantContent.push(...blocks)
      } else if (msg.role === "system" || msg.role === "developer") {
        // System/developer messages are handled via instructions; skip
      }
      continue
    }

    if (itemType === "function_call") {
      // Assistant made a tool call
      flushUser()
      const call = item as ResponseFunctionToolCallItem
      pendingAssistantContent.push({
        type: "tool_use",
        id: call.call_id,
        name: call.name,
        input: parseArguments(call.arguments),
      })
      continue
    }

    if (itemType === "function_call_output") {
      // User provided tool result
      flushAssistant()
      const output = item as ResponseFunctionCallOutputItem
      pendingUserContent.push({
        type: "tool_result",
        tool_use_id: output.call_id,
        content: stringifyOutput(output.output),
        is_error: output.status === "incomplete",
      })
      continue
    }

    if (itemType === "reasoning") {
      // Pass reasoning through as thinking blocks on assistant side
      flushUser()
      const reasoning = item as ResponseInputReasoning
      const thinkingText =
        reasoning.summary
          ?.map((s) => s.text)
          .filter(Boolean)
          .join("") || "Thinking..."
      pendingAssistantContent.push({
        type: "thinking",
        thinking: thinkingText,
        signature:
          reasoning.encrypted_content ?
            `${reasoning.encrypted_content}@${reasoning.id ?? ""}`
          : "",
      })
      continue
    }

    // Skip compaction, tool_search_call, tool_search_output, etc.
  }

  flushUser()
  flushAssistant()

  // Ensure we have at least one message
  if (messages.length === 0) {
    messages.push({ role: "user", content: "Hello" })
  }

  // Ensure messages alternate correctly - merge consecutive same-role if needed
  return ensureAlternatingMessages(messages)
}

const ensureAlternatingMessages = (
  messages: Array<AnthropicMessage>,
): Array<AnthropicMessage> => {
  if (messages.length <= 1) return messages

  const result: Array<AnthropicMessage> = [messages[0]]

  for (let i = 1; i < messages.length; i++) {
    const current = messages[i]
    const previous = result[result.length - 1]

    if (current.role === previous.role) {
      // Merge into previous
      if (current.role === "user") {
        const prev = previous as AnthropicUserMessage
        const curr = current
        const prevContent = normalizeUserContent(prev.content)
        const currContent = normalizeUserContent(curr.content)
        prev.content = [...prevContent, ...currContent]
      } else {
        const prev = previous as AnthropicAssistantMessage
        const curr = current
        const prevContent = normalizeAssistantContent(prev.content)
        const currContent = normalizeAssistantContent(curr.content)
        prev.content = [...prevContent, ...currContent]
      }
    } else {
      result.push(current)
    }
  }

  return result
}

const normalizeUserContent = (
  content: AnthropicUserMessage["content"],
): Array<AnthropicUserContentBlock> => {
  if (typeof content === "string") {
    return [{ type: "text", text: content }]
  }
  return content
}

const normalizeAssistantContent = (
  content: AnthropicAssistantMessage["content"],
): Array<AnthropicAssistantContentBlock> => {
  if (typeof content === "string") {
    return [{ type: "text", text: content }]
  }
  return content
}

const translateResponseContentToUserBlocks = (
  content: ResponseInputMessage["content"],
): Array<AnthropicUserContentBlock> => {
  if (!content) return [{ type: "text", text: "" }]
  if (typeof content === "string") {
    return [{ type: "text", text: content }]
  }

  const blocks: Array<AnthropicUserContentBlock> = []
  for (const item of content) {
    const itemType = (item as { type?: string }).type
    if (itemType === "input_text" || itemType === "output_text") {
      blocks.push({ type: "text", text: (item as ResponseInputText).text })
    } else if (itemType === "input_image") {
      const img = item as ResponseInputImage
      if (img.image_url) {
        const parsed = parseDataUrl(img.image_url)
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
    } else if (itemType === "input_file") {
      const file = item as ResponseInputFile
      if (file.file_data) {
        const parsed = parseDataUrl(file.file_data)
        if (parsed) {
          blocks.push({
            type: "document",
            source: {
              type: "base64",
              media_type: parsed.mediaType as "application/pdf",
              data: parsed.data,
            },
            title: file.filename ?? null,
          })
        }
      }
    }
  }

  return blocks.length > 0 ? blocks : [{ type: "text", text: "" }]
}

const translateResponseContentToAssistantBlocks = (
  content: ResponseInputMessage["content"],
): Array<AnthropicAssistantContentBlock> => {
  if (!content) return []
  if (typeof content === "string") {
    return content.length > 0 ? [{ type: "text", text: content }] : []
  }

  const blocks: Array<AnthropicAssistantContentBlock> = []
  for (const item of content) {
    const itemType = (item as { type?: string }).type
    if (itemType === "input_text" || itemType === "output_text") {
      const text = (item as ResponseInputText).text
      if (text.length > 0) {
        blocks.push({ type: "text", text })
      }
    }
  }

  return blocks
}

// ---------------------------------------------------------------------------
// Tool translation: Responses tools → Anthropic tools
// ---------------------------------------------------------------------------

const translateToolsToAnthropic = (
  tools: ResponsesPayload["tools"],
): Array<AnthropicTool> | undefined => {
  if (!tools || tools.length === 0) return undefined

  const result: Array<AnthropicTool> = []
  for (const tool of tools) {
    if (tool.type === "function") {
      const fn = tool as FunctionTool
      result.push({
        name: fn.name,
        description: fn.description ?? undefined,
        input_schema: fn.parameters ?? { type: "object", properties: {} },
      })
    }
    // Skip tool_search, namespace, web_search, etc. - not supported on messages API
  }

  return result.length > 0 ? result : undefined
}

const translateToolChoiceToAnthropic = (
  choice: ResponsesPayload["tool_choice"],
): AnthropicMessagesPayload["tool_choice"] | undefined => {
  if (!choice) return undefined

  if (typeof choice === "string") {
    switch (choice) {
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

  // ToolChoiceFunction: { type: "function", name: "..." }
  const fn = choice
  if (fn.type === "function" && fn.name) {
    return { type: "tool", name: fn.name }
  }

  return { type: "auto" }
}

// ---------------------------------------------------------------------------
// Response translation: Anthropic response → Responses result
// ---------------------------------------------------------------------------

export const translateAnthropicResultToResponses = (
  response: AnthropicResponse,
  requestModel: string,
): ResponsesResult => {
  const output: Array<ResponseOutputItem> = []
  let outputText = ""

  for (const block of response.content) {
    switch (block.type) {
      case "thinking": {
        const thinking = block
        const { encryptedContent, id } = parseSignature(thinking.signature)
        output.push({
          id: id || generateId(),
          type: "reasoning",
          summary:
            thinking.thinking && thinking.thinking !== "Thinking..." ?
              [{ type: "summary_text", text: thinking.thinking }]
            : [],
          encrypted_content: encryptedContent,
        } satisfies ResponseOutputReasoning)
        break
      }
      case "text": {
        const text = block.text
        outputText += text
        break
      }
      case "tool_use": {
        const toolUse = block
        output.push({
          id: generateId(),
          type: "function_call",
          call_id: toolUse.id,
          name: toolUse.name,
          arguments: JSON.stringify(toolUse.input),
          status: "completed",
        } satisfies ResponseOutputFunctionCall)
        break
      }
      default:
        break
    }
  }

  // If there was text output, wrap it in a message output item
  if (outputText.length > 0) {
    output.push({
      id: generateId(),
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: outputText }],
    } satisfies ResponseOutputMessage)
  }

  const status = mapAnthropicStopReasonToStatus(response.stop_reason)
  const incompleteDetails =
    response.stop_reason === "max_tokens" ?
      { reason: "max_output_tokens" as const }
    : null

  return {
    id: response.id,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    model: response.model || requestModel,
    output,
    output_text: outputText,
    status,
    usage: mapAnthropicUsageToResponses(response.usage),
    copilot_usage: response.copilot_usage ?? null,
    error: null,
    incomplete_details: incompleteDetails,
    instructions: null,
    metadata: null,
    parallel_tool_calls: true,
    temperature: null,
    tool_choice: "auto",
    tools: [],
    top_p: null,
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mapAnthropicStopReasonToStatus = (
  stopReason: AnthropicResponse["stop_reason"],
): string => {
  switch (stopReason) {
    case "end_turn":
    case "stop_sequence":
    case "tool_use":
      return "completed"
    case "max_tokens":
      return "incomplete"
    default:
      return "completed"
  }
}

const mapAnthropicUsageToResponses = (
  usage: AnthropicResponse["usage"],
): ResponsesResult["usage"] => {
  const inputTokens =
    (usage.input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0)
  return {
    input_tokens: inputTokens,
    output_tokens: usage.output_tokens ?? 0,
    input_tokens_details: {
      cached_tokens: usage.cache_read_input_tokens ?? 0,
    },
    output_tokens_details: {
      reasoning_tokens: 0,
    },
    total_tokens: inputTokens + (usage.output_tokens ?? 0),
  }
}

const parseSignature = (
  signature: string,
): { encryptedContent: string; id: string } => {
  if (!signature) return { encryptedContent: "", id: "" }
  const splitIndex = signature.lastIndexOf("@")
  if (splitIndex <= 0 || splitIndex === signature.length - 1) {
    return { encryptedContent: signature, id: "" }
  }
  return {
    encryptedContent: signature.slice(0, splitIndex),
    id: signature.slice(splitIndex + 1),
  }
}

const parseDataUrl = (
  dataUrl: string,
): { mediaType: string; data: string } | null => {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/)
  if (!match) return null
  return { mediaType: match[1], data: match[2] }
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

const stringifyOutput = (
  output: string | Array<ResponseInputContent>,
): string => {
  if (typeof output === "string") return output
  if (Array.isArray(output)) {
    return output
      .map((item) => {
        if ((item as ResponseInputText).type === "input_text") {
          return (item as ResponseInputText).text
        }
        if ((item as ResponseInputText).type === "output_text") {
          return (item as ResponseInputText).text
        }
        return ""
      })
      .join("")
  }
  return ""
}

let idCounter = 0
const generateId = (): string => {
  idCounter += 1
  return `resp_item_${Date.now().toString(36)}_${idCounter.toString(36)}`
}
