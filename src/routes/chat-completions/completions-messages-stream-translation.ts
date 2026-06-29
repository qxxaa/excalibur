/**
 * Translate Anthropic Messages API streaming events to Chat Completions
 * chunked streaming format (SSE with `chat.completion.chunk` objects).
 *
 * This is the stream counterpart of completions-messages-translation.ts.
 */

import type { AnthropicStreamEventData } from "~/routes/messages/anthropic-types"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CompletionsFromMessagesStreamState {
  responseId: string
  createdAt: number
  model: string
  roleSent: boolean
  toolCallIndex: number
}

interface SSEMessage {
  data?: string
  event?: string
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const createCompletionsFromMessagesStreamState =
  (): CompletionsFromMessagesStreamState => ({
    responseId: "",
    createdAt: Math.floor(Date.now() / 1000),
    model: "",
    roleSent: false,
    toolCallIndex: 0,
  })

/**
 * Translate a single Anthropic stream event into zero or more
 * Chat Completions chunk SSE messages.
 */
export const translateAnthropicStreamToCompletions = (
  event: AnthropicStreamEventData,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  switch (event.type) {
    case "message_start":
      return handleMessageStart(event, state)
    case "content_block_start":
      return handleContentBlockStart(event, state)
    case "content_block_delta":
      return handleContentBlockDelta(event, state)
    case "content_block_stop":
      return []
    case "message_delta":
      return handleMessageDelta(event, state)
    case "message_stop":
      return [{ data: "[DONE]" }]
    case "ping":
      return []
    case "error":
      return handleError(event, state)
    default:
      return []
  }
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

const handleMessageStart = (
  event: Extract<AnthropicStreamEventData, { type: "message_start" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  state.responseId = event.message.id
  state.model = event.message.model
  state.createdAt = Math.floor(Date.now() / 1000)

  // Emit initial role chunk
  state.roleSent = true
  return [
    createChunkMessage(state, {
      delta: { role: "assistant", content: "" },
      finishReason: null,
    }),
  ]
}

const handleContentBlockStart = (
  event: Extract<AnthropicStreamEventData, { type: "content_block_start" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  const block = event.content_block

  if (block.type === "tool_use") {
    const toolBlock = block as { type: "tool_use"; id: string; name: string }
    const messages: Array<SSEMessage> = [
      createChunkMessage(state, {
        delta: {
          tool_calls: [
            {
              index: state.toolCallIndex,
              id: toolBlock.id,
              type: "function",
              function: {
                name: toolBlock.name,
                arguments: "",
              },
            },
          ],
        },
        finishReason: null,
      }),
    ]
    return messages
  }

  // text and thinking blocks don't need a start event in completions format
  return []
}

const handleContentBlockDelta = (
  event: Extract<AnthropicStreamEventData, { type: "content_block_delta" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  const delta = event.delta

  if (delta.type === "text_delta") {
    return [
      createChunkMessage(state, {
        delta: { content: delta.text },
        finishReason: null,
      }),
    ]
  }

  if (delta.type === "input_json_delta") {
    return [
      createChunkMessage(state, {
        delta: {
          tool_calls: [
            {
              index: state.toolCallIndex,
              function: { arguments: delta.partial_json },
            },
          ],
        },
        finishReason: null,
      }),
    ]
  }

  if (delta.type === "thinking_delta") {
    // Map thinking to reasoning_content for clients that support it
    return [
      createChunkMessage(state, {
        delta: { reasoning_content: delta.thinking },
        finishReason: null,
      }),
    ]
  }

  // signature_delta: skip
  return []
}

const handleMessageDelta = (
  event: Extract<AnthropicStreamEventData, { type: "message_delta" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  const finishReason = mapStopReason(event.delta.stop_reason ?? null)
  const usage =
    event.usage ?
      {
        prompt_tokens: 0,
        completion_tokens: event.usage.output_tokens ?? 0,
        total_tokens: event.usage.output_tokens ?? 0,
      }
    : undefined

  return [
    createChunkMessage(state, {
      delta: {},
      finishReason,
      usage,
    }),
  ]
}

const handleError = (
  event: Extract<AnthropicStreamEventData, { type: "error" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  // Emit error as a final chunk with stop reason, then DONE
  return [
    createChunkMessage(state, {
      delta: { content: `[Error: ${event.error.message}]` },
      finishReason: "stop",
    }),
    { data: "[DONE]" },
  ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const createChunkMessage = (
  state: CompletionsFromMessagesStreamState,
  options: {
    delta: Record<string, unknown>
    finishReason?: "stop" | "length" | "tool_calls" | "content_filter" | null
    usage?: {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
    }
  },
): SSEMessage => {
  const chunk = {
    id: state.responseId || `chatcmpl-${Date.now().toString(36)}`,
    object: "chat.completion.chunk",
    created: state.createdAt,
    model: state.model,
    choices: [
      {
        index: 0,
        delta: options.delta,
        logprobs: null,
        finish_reason: options.finishReason ?? null,
      },
    ],
    ...(options.usage ? { usage: options.usage } : {}),
  }

  return { data: JSON.stringify(chunk) }
}

const mapStopReason = (
  stopReason: string | null,
): "stop" | "length" | "tool_calls" | "content_filter" | null => {
  switch (stopReason) {
    case "end_turn":
    case "stop_sequence":
      return "stop"
    case "max_tokens":
      return "length"
    case "tool_use":
      return "tool_calls"
    default:
      return stopReason ? "stop" : null
  }
}
