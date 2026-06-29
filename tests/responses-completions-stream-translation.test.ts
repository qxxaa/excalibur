/**
 * Tests for responses-completions-stream-translation.ts
 * (Chat Completions chunks -> Responses API stream events)
 */

import { describe, expect, it } from "bun:test"

import type { ChatCompletionChunk } from "~/services/copilot/create-chat-completions"

import {
  createResponsesFromCompletionsStreamState,
  translateCompletionsChunkToResponsesEvents,
} from "~/routes/responses/responses-completions-stream-translation"

describe("translateCompletionsChunkToResponsesEvents", () => {
  it("emits response.created and message item on first role chunk", () => {
    const state = createResponsesFromCompletionsStreamState()

    const chunk: ChatCompletionChunk = {
      id: "cmpl-1",
      object: "chat.completion.chunk",
      created: 1234567890,
      model: "gemini-3.5-flash",
      choices: [
        {
          index: 0,
          delta: { role: "assistant" },
          finish_reason: null,
          logprobs: null,
        },
      ],
    }

    const events = translateCompletionsChunkToResponsesEvents(chunk, state)

    const types = events.map((e) => e.type)
    expect(types).toContain("response.created")
    expect(types).toContain("response.in_progress")
    expect(types).toContain("response.output_item.added")
    expect(types).toContain("response.content_part.added")
  })

  it("emits output_text.delta for content chunks", () => {
    const state = createResponsesFromCompletionsStreamState()

    // First chunk - role
    translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    // Content delta
    const events = translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: { content: "Hello" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    expect(events).toHaveLength(1)
    expect(events[0].type).toBe("response.output_text.delta")
    expect(events[0].delta).toBe("Hello")
  })

  it("emits function_call events for tool call chunks", () => {
    const state = createResponsesFromCompletionsStreamState()

    // Role chunk
    translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: { role: "assistant" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    // Tool call start
    const startEvents = translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "tc_1",
                  type: "function",
                  function: { name: "search", arguments: "" },
                },
              ],
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    const addedEvent = startEvents.find(
      (e) => e.type === "response.output_item.added",
    )
    expect(addedEvent).toBeDefined()
    expect((addedEvent!.item as { type: string }).type).toBe("function_call")

    // Tool call arguments delta
    const argEvents = translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: '{"q":"test"}' },
                },
              ],
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    const argDelta = argEvents.find(
      (e) => e.type === "response.function_call_arguments.delta",
    )
    expect(argDelta).toBeDefined()
    expect(argDelta!.delta).toBe('{"q":"test"}')
  })

  it("emits response.completed on finish", () => {
    const state = createResponsesFromCompletionsStreamState()

    // Role + content
    translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Hi" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    // Finish
    const events = translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      },
      state,
    )

    const types = events.map((e) => e.type)
    expect(types).toContain("response.output_text.done")
    expect(types).toContain("response.output_item.done")
    expect(types).toContain("response.completed")

    const completed = events.find((e) => e.type === "response.completed")
    const response = completed!.response as {
      status: string
      output_text: string
      usage: { input_tokens: number }
    }
    expect(response.status).toBe("completed")
    expect(response.output_text).toBe("Hi")
    expect(response.usage.input_tokens).toBe(10)
  })

  it("emits incomplete status for length finish", () => {
    const state = createResponsesFromCompletionsStreamState()

    translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Trunc" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    const events = translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "length",
            logprobs: null,
          },
        ],
      },
      state,
    )

    const completed = events.find((e) => e.type === "response.completed")
    const response = completed!.response as {
      status: string
      incomplete_details: { reason: string }
    }
    expect(response.status).toBe("incomplete")
    expect(response.incomplete_details.reason).toBe("max_output_tokens")
  })

  it("handles late start (content without prior role chunk)", () => {
    const state = createResponsesFromCompletionsStreamState()

    // Content without prior role chunk
    const events = translateCompletionsChunkToResponsesEvents(
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1234567890,
        model: "gemini-3.5-flash",
        choices: [
          {
            index: 0,
            delta: { content: "Direct content" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      state,
    )

    const types = events.map((e) => e.type)
    // Should emit late start events + the delta
    expect(types).toContain("response.created")
    expect(types).toContain("response.output_item.added")
    expect(types).toContain("response.output_text.delta")
  })
})
