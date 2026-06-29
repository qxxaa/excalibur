import { describe, expect, it } from "bun:test"

import {
  createMessagesToResponsesStreamState,
  translateAnthropicStreamToResponsesEvent,
} from "../src/routes/responses/responses-messages-stream-translation"

import type { AnthropicStreamEventData } from "../src/routes/messages/anthropic-types"

describe("translateAnthropicStreamToResponsesEvent", () => {
  it("translates message_start to response.created", () => {
    const state = createMessagesToResponsesStreamState()
    const event: AnthropicStreamEventData = {
      type: "message_start",
      message: {
        id: "msg_001",
        type: "message",
        role: "assistant",
        content: [],
        model: "claude-sonnet-4-20250514",
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 100,
          output_tokens: 0,
        },
      },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.created")
    expect((result[0].response as { id: string }).id).toBe("msg_001")
    expect(state.responseCreatedSent).toBe(true)
  })

  it("translates text content_block_start to output_item.added", () => {
    const state = createMessagesToResponsesStreamState()
    const event: AnthropicStreamEventData = {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.output_item.added")
    expect((result[0].item as { type: string }).type).toBe("message")
    expect(state.blockTypes.get(0)).toBe("text")
  })

  it("translates thinking content_block_start to reasoning output_item.added", () => {
    const state = createMessagesToResponsesStreamState()
    const event: AnthropicStreamEventData = {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "" },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.output_item.added")
    expect((result[0].item as { type: string }).type).toBe("reasoning")
    expect(state.blockTypes.get(0)).toBe("thinking")
  })

  it("translates tool_use content_block_start to function_call output_item.added", () => {
    const state = createMessagesToResponsesStreamState()
    const event: AnthropicStreamEventData = {
      type: "content_block_start",
      index: 0,
      content_block: {
        type: "tool_use",
        id: "toolu_123",
        name: "search",
        input: {},
      },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.output_item.added")
    expect((result[0].item as { type: string }).type).toBe("function_call")
    expect((result[0].item as { call_id: string }).call_id).toBe("toolu_123")
    expect((result[0].item as { name: string }).name).toBe("search")
    expect(state.blockTypes.get(0)).toBe("tool_use")
  })

  it("translates text_delta to response.output_text.delta", () => {
    const state = createMessagesToResponsesStreamState()
    state.blockTypes.set(0, "text")

    const event: AnthropicStreamEventData = {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Hello" },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.output_text.delta")
    expect(result[0].delta).toBe("Hello")
  })

  it("translates thinking_delta to response.reasoning_summary_text.delta", () => {
    const state = createMessagesToResponsesStreamState()
    state.blockTypes.set(0, "thinking")

    const event: AnthropicStreamEventData = {
      type: "content_block_delta",
      index: 0,
      delta: { type: "thinking_delta", thinking: "Let me think..." },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.reasoning_summary_text.delta")
    expect(result[0].delta).toBe("Let me think...")
  })

  it("translates input_json_delta to function_call_arguments.delta", () => {
    const state = createMessagesToResponsesStreamState()
    state.blockTypes.set(0, "tool_use")

    const event: AnthropicStreamEventData = {
      type: "content_block_delta",
      index: 0,
      delta: { type: "input_json_delta", partial_json: '{"query":"wea' },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.function_call_arguments.delta")
    expect(result[0].delta).toBe('{"query":"wea')
  })

  it("translates content_block_stop for text to done events + output_item.done", () => {
    const state = createMessagesToResponsesStreamState()
    state.blockTypes.set(0, "text")

    const event: AnthropicStreamEventData = {
      type: "content_block_stop",
      index: 0,
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    // Should emit text done + output_item.done
    expect(result.length).toBeGreaterThanOrEqual(2)
    expect(result[0].type).toBe("response.output_text.done")
    expect(result[1].type).toBe("response.output_item.done")
    // outputIndex should increment
    expect(state.outputIndex).toBe(1)
  })

  it("translates content_block_stop for tool_use to done events", () => {
    const state = createMessagesToResponsesStreamState()
    state.blockTypes.set(0, "tool_use")
    state.toolCalls.set(0, { id: "toolu_abc", name: "search" })

    const event: AnthropicStreamEventData = {
      type: "content_block_stop",
      index: 0,
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result[0].type).toBe("response.function_call_arguments.done")
    expect(result[1].type).toBe("response.output_item.done")
    const doneItem = result[1].item as { type: string; call_id: string }
    expect(doneItem.type).toBe("function_call")
    expect(doneItem.call_id).toBe("toolu_abc")
  })

  it("translates message_stop to response.completed", () => {
    const state = createMessagesToResponsesStreamState()
    state.responseId = "msg_001"
    state.model = "claude-sonnet-4-20250514"
    state.createdAt = 1000

    const event: AnthropicStreamEventData = {
      type: "message_stop",
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.completed")
    expect((result[0].response as { status: string }).status).toBe("completed")
  })

  it("translates error event to response.failed", () => {
    const state = createMessagesToResponsesStreamState()

    const event: AnthropicStreamEventData = {
      type: "error",
      error: {
        type: "overloaded_error",
        message: "The server is overloaded",
      },
    }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(1)
    expect(result[0].type).toBe("response.failed")
    const response = result[0].response as { error: { message: string } }
    expect(response.error.message).toBe("The server is overloaded")
  })

  it("ignores ping events", () => {
    const state = createMessagesToResponsesStreamState()

    const event: AnthropicStreamEventData = { type: "ping" }

    const result = translateAnthropicStreamToResponsesEvent(event, state)

    expect(result).toHaveLength(0)
  })

  it("tracks message_delta usage", () => {
    const state = createMessagesToResponsesStreamState()

    const event: AnthropicStreamEventData = {
      type: "message_delta",
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 42 },
    }

    translateAnthropicStreamToResponsesEvent(event, state)

    expect(state.outputTokens).toBe(42)
  })

  it("handles full stream sequence", () => {
    const state = createMessagesToResponsesStreamState()

    // Simulate a full stream: message_start -> block_start -> deltas -> block_stop -> message_stop
    const events: Array<AnthropicStreamEventData> = [
      {
        type: "message_start",
        message: {
          id: "msg_full",
          type: "message",
          role: "assistant",
          content: [],
          model: "claude-sonnet-4-20250514",
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 50, output_tokens: 0 },
        },
      },
      {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hi there!" },
      },
      {
        type: "content_block_stop",
        index: 0,
      },
      {
        type: "message_delta",
        delta: { stop_reason: "end_turn" },
        usage: { output_tokens: 3 },
      },
      {
        type: "message_stop",
      },
    ]

    const allResults: Array<Record<string, unknown>> = []
    for (const event of events) {
      allResults.push(...translateAnthropicStreamToResponsesEvent(event, state))
    }

    // Should have: response.created, output_item.added, output_text.delta,
    // output_text.done, output_item.done, response.completed
    const types = allResults.map((r) => r.type)
    expect(types).toContain("response.created")
    expect(types).toContain("response.output_item.added")
    expect(types).toContain("response.output_text.delta")
    expect(types).toContain("response.output_text.done")
    expect(types).toContain("response.output_item.done")
    expect(types).toContain("response.completed")
  })
})
