/**
 * Translate Responses API payloads to Chat Completions payloads
 * and translate Chat Completions results back to Responses format.
 *
 * This is the reverse of completions-responses-translation.ts.
 * Used when a client sends /v1/responses but the model only supports
 * /chat/completions (e.g. Gemini on Copilot).
 *
 * Structured output: when the caller sends text.format.json_schema,
 * the schema is passed through as response_format.json_schema on the
 * completions side (which natively supports it for GPT/Gemini models).
 */

import type {
  ChatCompletionResponse,
  ChatCompletionsPayload,
  ContentPart,
  Message,
  Tool,
} from "~/services/copilot/create-chat-completions"
import type {
  FunctionTool,
  ResponseFunctionCallOutputItem,
  ResponseFunctionToolCallItem,
  ResponseInputMessage,
  ResponseOutputFunctionCall,
  ResponseOutputMessage,
  ResponsesPayload,
  ResponsesResult,
  ToolChoiceFunction,
  ToolChoiceOptions,
} from "~/services/copilot/create-responses"

// ---------------------------------------------------------------------------
// Payload: Responses -> Completions
// ---------------------------------------------------------------------------

export const translateResponsesToCompletionsPayload = (
  payload: ResponsesPayload,
): ChatCompletionsPayload => {
  const messages = translateInput(payload.instructions, payload.input)

  const completionsPayload: ChatCompletionsPayload = {
    model: payload.model,
    messages,
    stream: payload.stream,
    max_tokens: payload.max_output_tokens ?? undefined,
    top_p: payload.top_p,
    user: payload.metadata?.user_id,
    tools: translateTools(
      payload.tools as Array<Record<string, unknown>> | null | undefined,
    ),
    tool_choice: translateToolChoice(payload.tool_choice),
    parallel_tool_calls: payload.parallel_tool_calls ?? true,
    response_format: translateTextFormat(payload.text),
  }

  // Map reasoning effort
  const effort = (payload.reasoning as { effort?: string } | null)?.effort
  if (effort) {
    completionsPayload.reasoning_effort = effort
  }

  return completionsPayload
}

// ---------------------------------------------------------------------------
// Result: Completions -> Responses
// ---------------------------------------------------------------------------

export const translateCompletionsResultToResponses = (
  response: ChatCompletionResponse,
): ResponsesResult => {
  const choice = response.choices[0]
  const message = choice?.message

  const output: Array<ResponseOutputMessage | ResponseOutputFunctionCall> = []
  let outputText = ""

  // Text content
  if (message?.content) {
    outputText = message.content
    output.push({
      id: `msg_${response.id}`,
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: message.content }],
    })
  }

  // Tool calls
  if (message?.tool_calls) {
    for (const tc of message.tool_calls) {
      output.push({
        id: `fc_${tc.id}`,
        type: "function_call",
        call_id: tc.id,
        name: tc.function.name,
        arguments: tc.function.arguments,
        status: "completed",
      })
    }
  }

  // If no output at all, still emit an empty message
  if (output.length === 0) {
    output.push({
      id: `msg_${response.id}`,
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "" }],
    })
  }

  return {
    id: response.id,
    object: "response",
    created_at: response.created,
    model: response.model,
    output,
    output_text: outputText,
    status: getStatus(choice?.finish_reason, message?.tool_calls),
    usage: translateUsage(response.usage),
    ...(response.copilot_usage ?
      { copilot_usage: response.copilot_usage }
    : {}),
    error: null,
    incomplete_details: getIncompleteDetails(choice?.finish_reason),
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
// Input translation (Responses input -> Completions messages)
// ---------------------------------------------------------------------------

const translateInput = (
  instructions: string | null | undefined,
  input: string | Array<unknown> | undefined,
): Array<Message> => {
  const messages: Array<Message> = []

  // Instructions become system message
  if (instructions) {
    messages.push({ role: "system", content: instructions })
  }

  // Simple string input
  if (typeof input === "string") {
    messages.push({ role: "user", content: input })
    return messages
  }

  if (!Array.isArray(input)) {
    // No input at all - add a placeholder
    messages.push({ role: "user", content: "" })
    return messages
  }

  for (const item of input as Array<Record<string, unknown>>) {
    const type = item.type as string | undefined
    const role = item.role as string | undefined

    if (type === "function_call") {
      // Previous function call from assistant
      const fc = item as unknown as ResponseFunctionToolCallItem
      messages.push({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: fc.call_id,
            type: "function",
            function: {
              name: fc.name,
              arguments: fc.arguments,
            },
          },
        ],
      })
    } else if (type === "function_call_output") {
      // Tool result
      const fco = item as unknown as ResponseFunctionCallOutputItem
      messages.push({
        role: "tool",
        tool_call_id: fco.call_id,
        content:
          typeof fco.output === "string" ?
            fco.output
          : JSON.stringify(fco.output),
      })
    } else if (role === "user" || role === "assistant") {
      // Regular message
      const msg = item as unknown as ResponseInputMessage
      messages.push({
        role: msg.role as "user" | "assistant",
        content: translateInputContent(msg.content),
      })
    } else if (role === "system" || role === "developer") {
      // Additional system/developer messages
      const msg = item as unknown as ResponseInputMessage
      const text = extractTextFromContent(msg.content)
      if (text) {
        messages.push({ role: role, content: text })
      }
    }
    // Skip reasoning, compaction, tool_search items - no completions equivalent
  }

  // Ensure at least one message
  if (messages.length === 0 || messages.every((m) => m.role === "system")) {
    messages.push({ role: "user", content: "" })
  }

  return messages
}

// ---------------------------------------------------------------------------
// Content helpers
// ---------------------------------------------------------------------------

const translateInputContent = (
  content: ResponseInputMessage["content"],
): string | Array<ContentPart> | null => {
  if (typeof content === "string") {
    return content
  }
  if (!content || !Array.isArray(content) || content.length === 0) {
    return ""
  }

  const parts: Array<ContentPart> = []
  for (const block of content) {
    const type = (block as { type: string }).type
    if (type === "input_text" || type === "output_text") {
      parts.push({
        type: "text",
        text: (block as { text: string }).text,
      })
    } else if (type === "input_image") {
      const img = block as { image_url?: string; detail?: string }
      if (img.image_url) {
        parts.push({
          type: "image_url",
          image_url: {
            url: img.image_url,
            detail: (img.detail as "low" | "high" | "auto") ?? "auto",
          },
        })
      }
    } else if (type === "input_file") {
      const file = block as { file_data?: string; filename?: string }
      if (file.file_data) {
        parts.push({
          type: "file",
          file: {
            file_data: file.file_data,
            filename: file.filename,
          },
        })
      }
    }
  }

  if (parts.length === 0) return ""
  // Simplify single text part to string
  if (parts.length === 1 && parts[0].type === "text") {
    return (parts[0] as { text: string }).text
  }
  return parts
}

const extractTextFromContent = (
  content: ResponseInputMessage["content"],
): string | null => {
  if (typeof content === "string") return content
  if (!content || !Array.isArray(content)) return null
  const texts = (content as Array<{ type: string; text?: string }>)
    .filter(
      (b) => (b.type === "input_text" || b.type === "output_text") && b.text,
    )
    .map((b) => b.text!)
  return texts.length > 0 ? texts.join("\n\n") : null
}

// ---------------------------------------------------------------------------
// Tools translation
// ---------------------------------------------------------------------------

const translateTools = (
  tools: Array<Record<string, unknown>> | null | undefined,
): Array<Tool> | undefined => {
  if (!tools || tools.length === 0) return undefined

  const result: Array<Tool> = []
  for (const tool of tools) {
    if (tool.type === "function") {
      const ft = tool as unknown as FunctionTool
      result.push({
        type: "function",
        function: {
          name: ft.name,
          description: ft.description ?? undefined,
          parameters: ft.parameters ?? { type: "object", properties: {} },
        },
      })
    }
    // Skip non-function tools (web_search, tool_search, etc.) - no completions equivalent
  }

  return result.length > 0 ? result : undefined
}

const translateToolChoice = (
  toolChoice: ToolChoiceOptions | ToolChoiceFunction | undefined,
):
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } }
  | undefined => {
  if (!toolChoice) return undefined

  if (typeof toolChoice === "string") {
    if (
      toolChoice === "none"
      || toolChoice === "auto"
      || toolChoice === "required"
    ) {
      return toolChoice
    }
    return undefined
  }

  // { type: "function", name: "..." }
  const fc = toolChoice
  if (fc.type === "function" && fc.name) {
    return { type: "function", function: { name: fc.name } }
  }

  return undefined
}

// ---------------------------------------------------------------------------
// text.format -> response_format
// ---------------------------------------------------------------------------

interface TextFormat {
  format?: {
    type: string
    name?: string
    schema?: Record<string, unknown>
    strict?: boolean
  }
}

const translateTextFormat = (
  text: unknown,
): ChatCompletionsPayload["response_format"] => {
  if (!text || typeof text !== "object") return undefined

  const format = (text as TextFormat).format
  if (!format) return undefined

  if (format.type === "json_object") {
    return { type: "json_object" }
  }

  if (format.type === "json_schema" && format.schema) {
    return {
      type: "json_schema",
      json_schema: {
        name: format.name ?? "response",
        schema: format.schema,
        strict: format.strict,
      },
    }
  }

  return undefined
}

// ---------------------------------------------------------------------------
// Usage translation
// ---------------------------------------------------------------------------

const translateUsage = (
  usage: ChatCompletionResponse["usage"],
): ResponsesResult["usage"] => {
  if (!usage) return null

  return {
    input_tokens: usage.prompt_tokens,
    output_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens,
    ...(usage.prompt_tokens_details?.cached_tokens !== undefined ?
      {
        input_tokens_details: {
          cached_tokens: usage.prompt_tokens_details.cached_tokens,
        },
      }
    : {}),
  }
}

// ---------------------------------------------------------------------------
// Status and incomplete details
// ---------------------------------------------------------------------------

const getStatus = (
  finishReason: string | undefined,
  toolCalls: Array<unknown> | undefined,
): string => {
  if (toolCalls && toolCalls.length > 0) return "completed"
  if (finishReason === "length") return "incomplete"
  if (finishReason === "content_filter") return "incomplete"
  return "completed"
}

const getIncompleteDetails = (
  finishReason: string | undefined,
): ResponsesResult["incomplete_details"] => {
  if (finishReason === "length") {
    return { reason: "max_output_tokens" }
  }
  if (finishReason === "content_filter") {
    return { reason: "content_filter" }
  }
  return null
}
