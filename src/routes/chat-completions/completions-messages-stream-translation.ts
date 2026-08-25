/**
 * Translate Anthropic Messages stream events to Chat Completions chunks.
 *
 * Uses per-block state tracking (mirroring upstream's messages-stream-translation.ts
 * pattern) to correctly handle interleaved content blocks, tool calls, and
 * the synthetic forced tool_use used for structured output enforcement.
 *
 * When a forcedToolName is set, tool_use blocks matching that name are
 * remapped to text output and input_json_delta events become content deltas.
 */

import type { AnthropicStreamEventData } from "~/lib/types/anthropic"
import type {
  ChatCompletionChunk,
  CopilotUsage,
} from "~/lib/types/chat-completions"

// ---------------------------------------------------------------------------
// Block state - typed per block kind (mirrors upstream pattern)
// ---------------------------------------------------------------------------

interface StreamBlockBase {
  blockIndex: number
  done: boolean
}

interface StreamTextBlock extends StreamBlockBase {
  type: "text"
  text: string
}

interface StreamToolUseBlock extends StreamBlockBase {
  type: "tool_use"
  toolCallIndex: number
  id: string
  name: string
}

interface StreamForcedToolBlock extends StreamBlockBase {
  type: "forced_tool"
  text: string
}

interface StreamThinkingBlock extends StreamBlockBase {
  type: "thinking"
}

type StreamBlockState =
  | StreamTextBlock
  | StreamToolUseBlock
  | StreamForcedToolBlock
  | StreamThinkingBlock

// ---------------------------------------------------------------------------
// Translation state
// ---------------------------------------------------------------------------

interface CompletionsFromMessagesStreamState {
  responseId: string
  createdAt: number
  model: string
  roleSent: boolean
  nextToolCallIndex: number
  forcedToolName: string | null
  blocks: Map<number, StreamBlockState>
}

interface SSEMessage {
  data?: string
  event?: string
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const createCompletionsFromMessagesStreamState = (
  forcedToolName?: string,
): CompletionsFromMessagesStreamState => ({
  responseId: "",
  createdAt: Math.floor(Date.now() / 1000),
  model: "",
  roleSent: false,
  nextToolCallIndex: 0,
  forcedToolName: forcedToolName ?? null,
  blocks: new Map(),
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
    case "message_start": {
      state.responseId = event.message.id
      state.model = event.message.model
      return []
    }

    case "content_block_start": {
      return handleContentBlockStart(event, state)
    }

    case "content_block_delta": {
      return handleContentBlockDelta(event, state)
    }

    case "content_block_stop": {
      const block = state.blocks.get(event.index)
      if (block) {
        block.done = true
      }
      return []
    }

    case "message_delta": {
      const stopReason = event.delta.stop_reason
      const hasFunctionCalls = Array.from(state.blocks.values()).some(
        (b) => b.type === "tool_use",
      )

      // If we unwrapped a forced tool and there are no real tool calls,
      // the finish reason should be "stop", not "tool_calls"
      const suppressToolStop =
        state.forcedToolName !== null && !hasFunctionCalls
      const finishReason =
        suppressToolStop ? "stop"
        : hasFunctionCalls && stopReason === "tool_use" ? "tool_calls"
        : mapStopReason(stopReason)

      const messages: Array<SSEMessage> = [
        createChunkMessage(state, {
          delta: {},
          finishReason,
          usage:
            event.usage ?
              {
                prompt_tokens: event.usage.input_tokens ?? 0,
                completion_tokens: event.usage.output_tokens ?? 0,
                total_tokens:
                  (event.usage.input_tokens ?? 0)
                  + (event.usage.output_tokens ?? 0),
              }
            : undefined,
          copilotUsage: (event as { copilot_usage?: CopilotUsage })
            .copilot_usage,
        }),
      ]
      messages.push({ data: "[DONE]" })
      return messages
    }

    default:
      return []
  }
}

// ---------------------------------------------------------------------------
// Block handlers
// ---------------------------------------------------------------------------

const handleContentBlockStart = (
  event: Extract<AnthropicStreamEventData, { type: "content_block_start" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  const block = event.content_block
  const blockIndex = event.index

  if (block.type === "tool_use") {
    const toolBlock = block as { type: "tool_use"; id: string; name: string }

    // Synthetic forced tool for structured output - treat as text
    if (state.forcedToolName && toolBlock.name === state.forcedToolName) {
      state.blocks.set(blockIndex, {
        type: "forced_tool",
        blockIndex,
        text: "",
        done: false,
      })
      return []
    }

    // Real tool call
    const toolCallIndex = state.nextToolCallIndex++
    state.blocks.set(blockIndex, {
      type: "tool_use",
      blockIndex,
      toolCallIndex,
      id: toolBlock.id,
      name: toolBlock.name,
      done: false,
    })

    emitRoleChunkIfNeeded(state)
    return [
      createChunkMessage(state, {
        delta: {
          role: "assistant",
          tool_calls: [
            {
              index: toolCallIndex,
              id: toolBlock.id,
              type: "function",
              function: {
                name: toolBlock.name,
                arguments: "",
              },
            },
          ],
        },
      }),
    ]
  }

  if (block.type === "text") {
    state.blocks.set(blockIndex, {
      type: "text",
      blockIndex,
      text: block.text ?? "",
      done: false,
    })
    return []
  }

  if (block.type === "thinking") {
    state.blocks.set(blockIndex, {
      type: "thinking",
      blockIndex,
      done: false,
    })
    return []
  }

  return []
}

const handleContentBlockDelta = (
  event: Extract<AnthropicStreamEventData, { type: "content_block_delta" }>,
  state: CompletionsFromMessagesStreamState,
): Array<SSEMessage> => {
  const delta = event.delta
  const block = state.blocks.get(event.index)

  if (delta.type === "text_delta") {
    if (block?.type === "text") {
      block.text += delta.text
    }
    emitRoleChunkIfNeeded(state)
    return [
      createChunkMessage(state, {
        delta: { content: delta.text },
        finishReason: null,
      }),
    ]
  }

  if (delta.type === "input_json_delta") {
    if (!block) return []

    // Forced tool unwrapping: emit tool input as text content
    if (block.type === "forced_tool") {
      block.text += delta.partial_json
      emitRoleChunkIfNeeded(state)
      return [
        createChunkMessage(state, {
          delta: { content: delta.partial_json },
          finishReason: null,
        }),
      ]
    }

    // Real tool call argument delta
    if (block.type === "tool_use") {
      return [
        createChunkMessage(state, {
          delta: {
            tool_calls: [
              {
                index: block.toolCallIndex,
                type: "function",
                function: {
                  arguments: delta.partial_json,
                },
              },
            ],
          },
        }),
      ]
    }
  }

  // thinking_delta, signature_delta etc. - no completions equivalent
  return []
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const emitRoleChunkIfNeeded = (
  state: CompletionsFromMessagesStreamState,
): void => {
  // Role chunk is implicit in completions - no separate emission needed
  // when content follows. Tracked for consistency but not emitted separately.
  state.roleSent = true
}

const mapStopReason = (
  stopReason: string | null | undefined,
): "stop" | "length" | "content_filter" => {
  if (stopReason === "max_tokens") return "length"
  if (stopReason === "content_filter") return "content_filter"
  return "stop"
}

interface CreateChunkOptions {
  delta: ChatCompletionChunk["choices"][0]["delta"]
  finishReason?: ChatCompletionChunk["choices"][0]["finish_reason"]
  usage?: ChatCompletionChunk["usage"]
  copilotUsage?: CopilotUsage | null
}

const createChunkMessage = (
  state: CompletionsFromMessagesStreamState,
  { delta, finishReason = null, usage, copilotUsage }: CreateChunkOptions,
): SSEMessage => {
  const chunk: ChatCompletionChunk = {
    id: state.responseId,
    object: "chat.completion.chunk",
    created: state.createdAt,
    model: state.model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
        logprobs: null,
      },
    ],
    ...(usage ? { usage } : {}),
    ...(copilotUsage ? { copilot_usage: copilotUsage } : {}),
  }

  return { data: JSON.stringify(chunk) }
}
