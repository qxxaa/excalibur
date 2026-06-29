/**
 * Anthropic Messages stream → Responses API stream translation.
 *
 * When a /v1/responses request is fulfilled via the Messages API,
 * this module translates the incoming Anthropic SSE events into
 * Responses API SSE events that the caller expects.
 *
 * This is the reverse direction of messages/responses-stream-translation.ts.
 */

import type {
  AnthropicContentBlockDeltaEvent,
  AnthropicContentBlockStartEvent,
  AnthropicContentBlockStopEvent,
  AnthropicErrorEvent,
  AnthropicMessageDeltaEvent,
  AnthropicMessageStartEvent,
  AnthropicStreamEventData,
} from "~/routes/messages/anthropic-types"

import type { ResponsesResult } from "~/services/copilot/create-responses"

// ---------------------------------------------------------------------------
// Stream state
// ---------------------------------------------------------------------------

export interface MessagesToResponsesStreamState {
  responseId: string
  model: string
  createdAt: number
  outputIndex: number
  contentIndex: number
  sequenceNumber: number
  // Track what type each anthropic block index maps to
  blockTypes: Map<number, "text" | "thinking" | "tool_use">
  // Track tool call info per anthropic block index
  toolCalls: Map<number, { id: string; name: string }>
  responseCreatedSent: boolean
  inputTokens: number
  outputTokens: number
  cachedTokens: number
  // Name of synthetic forced tool for structured output unwrapping
  forcedToolName: string | null
}

export const createMessagesToResponsesStreamState = (
  forcedToolName?: string,
): MessagesToResponsesStreamState => ({
  responseId: "",
  model: "",
  createdAt: 0,
  outputIndex: 0,
  contentIndex: 0,
  sequenceNumber: 0,
  blockTypes: new Map(),
  toolCalls: new Map(),
  responseCreatedSent: false,
  inputTokens: 0,
  outputTokens: 0,
  cachedTokens: 0,
  forcedToolName: forcedToolName ?? null,
})

// ---------------------------------------------------------------------------
// Main translation function
// ---------------------------------------------------------------------------

export const translateAnthropicStreamToResponsesEvent = (
  event: AnthropicStreamEventData,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  switch (event.type) {
    case "message_start":
      return handleMessageStart(event, state)
    case "content_block_start":
      return handleContentBlockStart(event, state)
    case "content_block_delta":
      return handleContentBlockDelta(event, state)
    case "content_block_stop":
      return handleContentBlockStop(event, state)
    case "message_delta":
      return handleMessageDelta(event, state)
    case "message_stop":
      return handleMessageStop(state)
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
  event: AnthropicMessageStartEvent,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  const message = event.message
  state.responseId = message.id
  state.model = message.model
  state.createdAt = Math.floor(Date.now() / 1000)
  state.inputTokens = message.usage?.input_tokens ?? 0
  state.cachedTokens =
    (message.usage as { cache_read_input_tokens?: number })
      ?.cache_read_input_tokens ?? 0

  const events: Array<Record<string, unknown>> = []

  if (!state.responseCreatedSent) {
    events.push({
      type: "response.created",
      response: buildPartialResponse(state),
    })
    state.responseCreatedSent = true
  }

  return events
}

const handleContentBlockStart = (
  event: AnthropicContentBlockStartEvent,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  const events: Array<Record<string, unknown>> = []
  const block = event.content_block
  const blockIndex = event.index

  if (block.type === "text") {
    state.blockTypes.set(blockIndex, "text")
    // Emit output_item.added for a message item with text content
    events.push({
      type: "response.output_item.added",
      output_index: state.outputIndex,
      item: {
        id: `msg_${state.outputIndex}`,
        type: "message",
        role: "assistant",
        status: "in_progress",
        content: [{ type: "output_text", text: "" }],
      },
    })
  } else if (block.type === "thinking") {
    state.blockTypes.set(blockIndex, "thinking")
    events.push({
      type: "response.output_item.added",
      output_index: state.outputIndex,
      item: {
        id: `reasoning_${state.outputIndex}`,
        type: "reasoning",
        summary: [],
      },
    })
  } else if (block.type === "tool_use") {
    const toolBlock = block as { type: "tool_use"; id: string; name: string }
    if (state.forcedToolName && toolBlock.name === state.forcedToolName) {
      // Synthetic tool for structured output - treat as text
      state.blockTypes.set(blockIndex, "text")
      events.push({
        type: "response.output_item.added",
        output_index: state.outputIndex,
        item: {
          id: `msg_${state.outputIndex}`,
          type: "message",
          role: "assistant",
          status: "in_progress",
          content: [{ type: "output_text", text: "" }],
        },
      })
    } else {
      state.blockTypes.set(blockIndex, "tool_use")
      state.toolCalls.set(blockIndex, {
        id: toolBlock.id,
        name: toolBlock.name,
      })
      events.push({
        type: "response.output_item.added",
        output_index: state.outputIndex,
        item: {
          id: `fc_${state.outputIndex}`,
          type: "function_call",
          call_id: toolBlock.id,
          name: toolBlock.name,
          arguments: "",
          status: "in_progress",
        },
      })
    }
  }

  return events
}

const handleContentBlockDelta = (
  event: AnthropicContentBlockDeltaEvent,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  const events: Array<Record<string, unknown>> = []
  const blockIndex = event.index
  const delta = event.delta
  const blockType = state.blockTypes.get(blockIndex)

  if (delta.type === "text_delta" && blockType === "text") {
    events.push({
      type: "response.output_text.delta",
      output_index: state.outputIndex,
      content_index: 0,
      delta: delta.text,
    })
  } else if (delta.type === "thinking_delta" && blockType === "thinking") {
    events.push({
      type: "response.reasoning_summary_text.delta",
      output_index: state.outputIndex,
      summary_index: 0,
      delta: delta.thinking,
    })
  } else if (delta.type === "input_json_delta" && blockType === "tool_use") {
    events.push({
      type: "response.function_call_arguments.delta",
      output_index: state.outputIndex,
      delta: delta.partial_json,
    })
  } else if (delta.type === "input_json_delta" && blockType === "text") {
    // Synthetic tool unwrapping: emit tool input as text content
    events.push({
      type: "response.output_text.delta",
      output_index: state.outputIndex,
      content_index: 0,
      delta: delta.partial_json,
    })
  }
  // signature_delta: skip for now (reasoning signatures are opaque to callers)

  return events
}

const handleContentBlockStop = (
  event: AnthropicContentBlockStopEvent,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  const events: Array<Record<string, unknown>> = []
  const blockIndex = event.index
  const blockType = state.blockTypes.get(blockIndex)

  if (blockType === "text") {
    events.push({
      type: "response.output_text.done",
      output_index: state.outputIndex,
      content_index: 0,
      text: "",
    })
  } else if (blockType === "thinking") {
    events.push({
      type: "response.reasoning_summary_text.done",
      output_index: state.outputIndex,
      summary_index: 0,
      text: "",
    })
  } else if (blockType === "tool_use") {
    events.push({
      type: "response.function_call_arguments.done",
      output_index: state.outputIndex,
      arguments: "",
    })
  }

  // Emit output_item.done
  events.push({
    type: "response.output_item.done",
    output_index: state.outputIndex,
    item: buildOutputItemDone(blockType, blockIndex, state),
  })

  state.outputIndex += 1
  state.blockTypes.delete(blockIndex)

  return events
}

const handleMessageDelta = (
  event: AnthropicMessageDeltaEvent,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  if (event.usage?.output_tokens) {
    state.outputTokens = event.usage.output_tokens
  }
  // The actual completion event is handled in message_stop
  return []
}

const handleMessageStop = (
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  return [
    {
      type: "response.completed",
      response: buildCompletedResponse(state),
    },
  ]
}

const handleError = (
  event: AnthropicErrorEvent,
  state: MessagesToResponsesStreamState,
): Array<Record<string, unknown>> => {
  return [
    {
      type: "response.failed",
      response: {
        ...buildPartialResponse(state),
        status: "failed",
        error: {
          code: event.error.type ?? "api_error",
          message: event.error.message,
        },
      },
    },
  ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const buildOutputItemDone = (
  blockType: string | undefined,
  blockIndex: number,
  state: MessagesToResponsesStreamState,
): Record<string, unknown> => {
  if (blockType === "tool_use") {
    const toolInfo = state.toolCalls.get(blockIndex)
    return {
      id: `fc_${state.outputIndex}`,
      type: "function_call",
      call_id: toolInfo?.id ?? "",
      name: toolInfo?.name ?? "",
      arguments: "",
      status: "completed",
    }
  }

  if (blockType === "thinking") {
    return {
      id: `reasoning_${state.outputIndex}`,
      type: "reasoning",
      summary: [],
    }
  }

  return {
    id: `msg_${state.outputIndex}`,
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text: "" }],
  }
}

const buildPartialResponse = (
  state: MessagesToResponsesStreamState,
): ResponsesResult => ({
  id: state.responseId || `resp_${Date.now().toString(36)}`,
  object: "response",
  created_at: state.createdAt || Math.floor(Date.now() / 1000),
  model: state.model,
  output: [],
  output_text: "",
  status: "in_progress",
  usage: {
    input_tokens: state.inputTokens + state.cachedTokens,
    output_tokens: state.outputTokens,
    input_tokens_details: { cached_tokens: state.cachedTokens },
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: state.inputTokens + state.cachedTokens + state.outputTokens,
  },
  copilot_usage: null,
  error: null,
  incomplete_details: null,
  instructions: null,
  metadata: null,
  parallel_tool_calls: true,
  temperature: null,
  tool_choice: "auto",
  tools: [],
  top_p: null,
})

const buildCompletedResponse = (
  state: MessagesToResponsesStreamState,
): ResponsesResult => ({
  ...buildPartialResponse(state),
  status: "completed",
})
