/**
 * Translate Chat Completions streaming chunks back to Responses API
 * streaming events (SSE with response.* event types).
 *
 * This is the reverse of completions-responses-stream-translation.ts.
 * Used when a client sends /v1/responses (streaming) but the model
 * only supports /chat/completions.
 */

import type { ChatCompletionChunk } from "~/services/copilot/create-chat-completions"
import type {
  ResponseOutputFunctionCall,
  ResponseOutputMessage,
  ResponsesResult,
  ResponseUsage,
} from "~/services/copilot/create-responses"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ResponsesFromCompletionsStreamState {
  responseId: string
  createdAt: number
  model: string
  messageOutputEmitted: boolean
  outputText: string
  sequenceNumber: number
  toolCalls: Map<number, { callId: string; name: string; arguments: string }>
}

// Loose event record - we build events dynamically
type ResponseEvent = Record<string, unknown>

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const createResponsesFromCompletionsStreamState =
  (): ResponsesFromCompletionsStreamState => ({
    responseId: "",
    createdAt: Math.floor(Date.now() / 1000),
    model: "",
    messageOutputEmitted: false,
    outputText: "",
    sequenceNumber: 0,
    toolCalls: new Map(),
  })

/**
 * Translate a single Chat Completions chunk into zero or more
 * Responses API stream events.
 */
export const translateCompletionsChunkToResponsesEvents = (
  chunk: ChatCompletionChunk,
  state: ResponsesFromCompletionsStreamState,
): Array<ResponseEvent> => {
  const events: Array<ResponseEvent> = []

  // Capture metadata
  if (chunk.id) state.responseId = chunk.id
  if (chunk.created) state.createdAt = chunk.created
  if (chunk.model) state.model = chunk.model

  const choice = chunk.choices?.[0]
  if (!choice) {
    // No choices - might be a usage-only chunk at the end
    return events
  }

  const delta = choice.delta
  const finishReason = choice.finish_reason

  // Emit response.created on first chunk with role
  if (delta?.role === "assistant" && !state.messageOutputEmitted) {
    events.push({
      type: "response.created",
      response: buildPartialResponse(state, "in_progress"),
      sequence_number: state.sequenceNumber++,
    })
    events.push({
      type: "response.in_progress",
      response: buildPartialResponse(state, "in_progress"),
      sequence_number: state.sequenceNumber++,
    })
    // Emit message output item
    state.messageOutputEmitted = true
    events.push({
      type: "response.output_item.added",
      output_index: 0,
      item: {
        id: `msg_${state.responseId}`,
        type: "message",
        role: "assistant",
        status: "in_progress",
        content: [],
      },
      sequence_number: state.sequenceNumber++,
    })
    // Emit content part added
    events.push({
      type: "response.content_part.added",
      item_id: `msg_${state.responseId}`,
      output_index: 0,
      content_index: 0,
      part: { type: "output_text", text: "" },
      sequence_number: state.sequenceNumber++,
    })
  }

  // Text content delta
  if (delta?.content) {
    if (!state.messageOutputEmitted) {
      // Late start - emit setup events
      emitLateStart(state, events)
    }
    state.outputText += delta.content
    events.push({
      type: "response.output_text.delta",
      item_id: `msg_${state.responseId}`,
      output_index: 0,
      content_index: 0,
      delta: delta.content,
      sequence_number: state.sequenceNumber++,
    })
  }

  // Tool call deltas
  if (delta?.tool_calls) {
    for (const tc of delta.tool_calls) {
      const idx = tc.index ?? 0
      if (tc.id && tc.function?.name) {
        // New tool call start
        state.toolCalls.set(idx, {
          callId: tc.id,
          name: tc.function.name,
          arguments: tc.function.arguments ?? "",
        })
        // Calculate the output_index: message is 0, tool calls start at 1
        const outputIndex = state.toolCalls.size
        events.push({
          type: "response.output_item.added",
          output_index: outputIndex,
          item: {
            id: `fc_${tc.id}`,
            type: "function_call",
            call_id: tc.id,
            name: tc.function.name,
            arguments: "",
            status: "in_progress",
          },
          sequence_number: state.sequenceNumber++,
        })
      }
      if (tc.function?.arguments) {
        const existing = state.toolCalls.get(idx)
        if (existing) {
          existing.arguments += tc.function.arguments
        }
        const outputIndex = idx + 1
        events.push({
          type: "response.function_call_arguments.delta",
          item_id: `fc_${state.toolCalls.get(idx)?.callId ?? "unknown"}`,
          output_index: outputIndex,
          delta: tc.function.arguments,
          sequence_number: state.sequenceNumber++,
        })
      }
    }
  }

  // Finish
  if (finishReason) {
    // Close text content
    if (state.messageOutputEmitted) {
      events.push({
        type: "response.output_text.done",
        item_id: `msg_${state.responseId}`,
        output_index: 0,
        content_index: 0,
        text: state.outputText,
        sequence_number: state.sequenceNumber++,
      })
      events.push({
        type: "response.output_item.done",
        output_index: 0,
        item: {
          id: `msg_${state.responseId}`,
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: state.outputText }],
        },
        sequence_number: state.sequenceNumber++,
      })
    }

    // Close tool calls
    for (const [idx, tc] of state.toolCalls) {
      const outputIndex = idx + 1
      events.push({
        type: "response.function_call_arguments.done",
        item_id: `fc_${tc.callId}`,
        output_index: outputIndex,
        arguments: tc.arguments,
        sequence_number: state.sequenceNumber++,
      })
      events.push({
        type: "response.output_item.done",
        output_index: outputIndex,
        item: {
          id: `fc_${tc.callId}`,
          type: "function_call",
          call_id: tc.callId,
          name: tc.name,
          arguments: tc.arguments,
          status: "completed",
        },
        sequence_number: state.sequenceNumber++,
      })
    }

    // Build final response
    const output: Array<ResponseOutputMessage | ResponseOutputFunctionCall> = []
    if (state.messageOutputEmitted) {
      output.push({
        id: `msg_${state.responseId}`,
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: state.outputText }],
      })
    }
    for (const [, tc] of state.toolCalls) {
      output.push({
        id: `fc_${tc.callId}`,
        type: "function_call",
        call_id: tc.callId,
        name: tc.name,
        arguments: tc.arguments,
        status: "completed",
      })
    }

    const usage: ResponseUsage | null = translateChunkUsage(chunk.usage)

    events.push({
      type: "response.completed",
      response: {
        id: state.responseId,
        object: "response",
        created_at: state.createdAt,
        model: state.model,
        output,
        output_text: state.outputText,
        status: finishReason === "length" ? "incomplete" : "completed",
        usage,
        error: null,
        incomplete_details:
          finishReason === "length" ? { reason: "max_output_tokens" }
          : finishReason === "content_filter" ? { reason: "content_filter" }
          : null,
        instructions: null,
        metadata: null,
        parallel_tool_calls: true,
        temperature: null,
        tool_choice: "auto",
        tools: [],
        top_p: null,
      } satisfies ResponsesResult,
      ...(chunk.copilot_usage ? { copilot_usage: chunk.copilot_usage } : {}),
      sequence_number: state.sequenceNumber++,
    })
  }

  return events
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const emitLateStart = (
  state: ResponsesFromCompletionsStreamState,
  events: Array<ResponseEvent>,
): void => {
  state.messageOutputEmitted = true
  events.push({
    type: "response.created",
    response: buildPartialResponse(state, "in_progress"),
    sequence_number: state.sequenceNumber++,
  })
  events.push({
    type: "response.in_progress",
    response: buildPartialResponse(state, "in_progress"),
    sequence_number: state.sequenceNumber++,
  })
  events.push({
    type: "response.output_item.added",
    output_index: 0,
    item: {
      id: `msg_${state.responseId}`,
      type: "message",
      role: "assistant",
      status: "in_progress",
      content: [],
    },
    sequence_number: state.sequenceNumber++,
  })
  events.push({
    type: "response.content_part.added",
    item_id: `msg_${state.responseId}`,
    output_index: 0,
    content_index: 0,
    part: { type: "output_text", text: "" },
    sequence_number: state.sequenceNumber++,
  })
}

const buildPartialResponse = (
  state: ResponsesFromCompletionsStreamState,
  status: string,
): Record<string, unknown> => ({
  id: state.responseId,
  object: "response",
  created_at: state.createdAt,
  model: state.model,
  output: [],
  output_text: "",
  status,
  usage: null,
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

const translateChunkUsage = (
  usage: ChatCompletionChunk["usage"],
): ResponseUsage | null => {
  if (!usage) return null
  return {
    input_tokens: usage.prompt_tokens ?? 0,
    output_tokens: usage.completion_tokens ?? 0,
    total_tokens: usage.total_tokens ?? 0,
    ...(usage.prompt_tokens_details?.cached_tokens !== undefined ?
      {
        input_tokens_details: {
          cached_tokens: usage.prompt_tokens_details.cached_tokens,
        },
      }
    : {}),
  }
}
