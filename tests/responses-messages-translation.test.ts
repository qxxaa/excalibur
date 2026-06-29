import { describe, expect, it } from "bun:test"

import {
  translateResponsesToMessagesPayload,
  translateAnthropicResultToResponses,
} from "../src/routes/responses/responses-messages-translation"

import type { ResponsesPayload } from "../src/services/copilot/create-responses"
import type { AnthropicResponse } from "../src/routes/messages/anthropic-types"

describe("translateResponsesToMessagesPayload", () => {
  it("translates a simple text input", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [
        {
          type: "message",
          role: "user",
          content: "Hello, Claude!",
        },
      ],
      max_output_tokens: 4096,
      stream: false,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.model).toBe("claude-sonnet-4-20250514")
    expect(result.max_tokens).toBe(4096)
    expect(result.stream).toBe(false)
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].role).toBe("user")
  })

  it("hoists instructions to system", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      instructions: "You are a helpful assistant.",
      input: [
        {
          type: "message",
          role: "user",
          content: "Hi",
        },
      ],
      max_output_tokens: 4096,
      stream: false,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.system).toBe("You are a helpful assistant.")
  })

  it("translates string input to user message", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: "What is 2+2?",
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].role).toBe("user")
    expect(result.messages[0].content).toBe("What is 2+2?")
  })

  it("translates function tools to anthropic format", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [{ type: "message", role: "user", content: "Search for news" }],
      tools: [
        {
          type: "function",
          name: "search",
          description: "Search the web",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
          strict: false,
        },
      ],
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].name).toBe("search")
    expect(result.tools![0].description).toBe("Search the web")
    expect(result.tools![0].input_schema).toEqual({
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    })
  })

  it("translates tool_choice string values", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [{ type: "message", role: "user", content: "Hi" }],
      tool_choice: "required",
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.tool_choice).toEqual({ type: "any" })
  })

  it("translates tool_choice function to tool type", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [{ type: "message", role: "user", content: "Hi" }],
      tool_choice: { type: "function", name: "search" },
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.tool_choice).toEqual({ type: "tool", name: "search" })
  })

  it("maps reasoning effort to thinking config", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [{ type: "message", role: "user", content: "Think hard" }],
      reasoning: { effort: "high" },
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.thinking).toBeDefined()
    expect(result.thinking!.type).toBe("enabled")
    expect(result.thinking!.budget_tokens).toBeGreaterThan(0)
  })

  it("handles multi-turn conversation with tool calls", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [
        { type: "message", role: "user", content: "Search for weather" },
        {
          type: "function_call",
          call_id: "call_123",
          name: "search",
          arguments: '{"query":"weather"}',
          status: "completed",
        },
        {
          type: "function_call_output",
          call_id: "call_123",
          output: "It's sunny today",
          status: "completed",
        },
      ],
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    // Should have: user message, assistant with tool_use, user with tool_result
    expect(result.messages.length).toBeGreaterThanOrEqual(2)

    // First message should be user
    expect(result.messages[0].role).toBe("user")

    // Second should be assistant with tool_use
    expect(result.messages[1].role).toBe("assistant")
    const assistantContent = result.messages[1].content as Array<{
      type: string
    }>
    expect(assistantContent.some((b) => b.type === "tool_use")).toBe(true)

    // Third should be user with tool_result
    expect(result.messages[2].role).toBe("user")
    const userContent = result.messages[2].content as Array<{ type: string }>
    expect(userContent.some((b) => b.type === "tool_result")).toBe(true)
  })

  it("handles content array with input_text blocks", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [
        {
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: "Hello" },
            { type: "input_text", text: " world" },
          ],
        },
      ],
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.messages).toHaveLength(1)
    const content = result.messages[0].content as Array<{
      type: string
      text: string
    }>
    expect(content).toHaveLength(2)
    expect(content[0].type).toBe("text")
    expect(content[0].text).toBe("Hello")
    expect(content[1].type).toBe("text")
    expect(content[1].text).toBe(" world")
  })

  it("skips tool_search and web_search tools", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [{ type: "message", role: "user", content: "Hi" }],
      tools: [
        { type: "web_search" },
        { type: "tool_search", execution: "client" },
        {
          type: "function",
          name: "valid_tool",
          parameters: { type: "object", properties: {} },
          strict: false,
        },
      ],
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].name).toBe("valid_tool")
  })

  it("omits system when instructions is null", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      instructions: null,
      input: [{ type: "message", role: "user", content: "Hi" }],
      max_output_tokens: 4096,
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.system).toBeUndefined()
  })

  it("defaults max_tokens to 8192 when not specified", () => {
    const payload: ResponsesPayload = {
      model: "claude-sonnet-4-20250514",
      input: [{ type: "message", role: "user", content: "Hi" }],
      store: false,
    }

    const result = translateResponsesToMessagesPayload(payload)

    expect(result.max_tokens).toBe(8192)
  })
})

describe("translateAnthropicResultToResponses", () => {
  it("translates a simple text response", () => {
    const response: AnthropicResponse = {
      id: "msg_123",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Hello! How can I help?" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 10,
        output_tokens: 8,
      },
    }

    const result = translateAnthropicResultToResponses(
      response,
      "claude-sonnet-4-20250514",
    )

    expect(result.id).toBe("msg_123")
    expect(result.object).toBe("response")
    expect(result.status).toBe("completed")
    expect(result.output_text).toBe("Hello! How can I help?")
    expect(result.output.length).toBeGreaterThan(0)

    // Should have a message output item
    const msgItem = result.output.find((o) => o.type === "message")
    expect(msgItem).toBeDefined()
  })

  it("translates tool_use blocks to function_call outputs", () => {
    const response: AnthropicResponse = {
      id: "msg_456",
      type: "message",
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "toolu_abc",
          name: "search",
          input: { query: "weather" },
        },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: {
        input_tokens: 20,
        output_tokens: 15,
      },
    }

    const result = translateAnthropicResultToResponses(
      response,
      "claude-sonnet-4-20250514",
    )

    expect(result.status).toBe("completed")
    const fnCall = result.output.find((o) => o.type === "function_call")
    expect(fnCall).toBeDefined()
    expect((fnCall as { call_id: string }).call_id).toBe("toolu_abc")
    expect((fnCall as { name: string }).name).toBe("search")
    expect((fnCall as { arguments: string }).arguments).toBe(
      '{"query":"weather"}',
    )
  })

  it("translates thinking blocks to reasoning output", () => {
    const response: AnthropicResponse = {
      id: "msg_789",
      type: "message",
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: "Let me analyze this...",
          signature: "enc_content_xyz@id_123",
        },
        { type: "text", text: "The answer is 42." },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 30,
        output_tokens: 25,
      },
    }

    const result = translateAnthropicResultToResponses(
      response,
      "claude-sonnet-4-20250514",
    )

    const reasoning = result.output.find((o) => o.type === "reasoning")
    expect(reasoning).toBeDefined()
    expect((reasoning as { encrypted_content: string }).encrypted_content).toBe(
      "enc_content_xyz",
    )

    expect(result.output_text).toBe("The answer is 42.")
  })

  it("maps max_tokens stop reason to incomplete status", () => {
    const response: AnthropicResponse = {
      id: "msg_trunc",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Partial response..." }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "max_tokens",
      stop_sequence: null,
      usage: {
        input_tokens: 10,
        output_tokens: 4096,
      },
    }

    const result = translateAnthropicResultToResponses(
      response,
      "claude-sonnet-4-20250514",
    )

    expect(result.status).toBe("incomplete")
    expect(result.incomplete_details).toEqual({ reason: "max_output_tokens" })
  })

  it("maps usage with cache_read_input_tokens", () => {
    const response: AnthropicResponse = {
      id: "msg_cache",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Hi" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 100,
        output_tokens: 5,
        cache_read_input_tokens: 80,
      },
    }

    const result = translateAnthropicResultToResponses(
      response,
      "claude-sonnet-4-20250514",
    )

    expect(result.usage!.input_tokens).toBe(180) // 100 + 80
    expect(result.usage!.input_tokens_details!.cached_tokens).toBe(80)
    expect(result.usage!.output_tokens).toBe(5)
    expect(result.usage!.total_tokens).toBe(185)
  })

  it("preserves copilot_usage", () => {
    const response: AnthropicResponse = {
      id: "msg_aiu",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Done" }],
      copilot_usage: { total_nano_aiu: 5000 },
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 10,
        output_tokens: 2,
      },
    }

    const result = translateAnthropicResultToResponses(
      response,
      "claude-sonnet-4-20250514",
    )

    expect(result.copilot_usage).toEqual({ total_nano_aiu: 5000 })
  })
})
