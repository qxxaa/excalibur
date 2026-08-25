import { describe, expect, it } from "bun:test"

import {
  translateCompletionsToMessagesPayload,
  translateAnthropicResultToCompletions,
} from "../src/routes/chat-completions/completions-messages-translation"

import type { ChatCompletionsPayload } from "~/lib/types/chat-completions"
import type { AnthropicResponse } from "~/lib/types/anthropic"

// ===========================================================================
// Request translation: Completions -> Messages
// ===========================================================================

describe("translateCompletionsToMessagesPayload", () => {
  describe("basic payload structure", () => {
    it("translates a minimal payload", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hello" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.model).toBe("claude-sonnet-4-20250514")
      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe("user")
      expect(result.max_tokens).toBe(8192)
    })

    it("uses max_completion_tokens over max_tokens", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 1000,
        max_completion_tokens: 2000,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.max_tokens).toBe(2000)
    })

    it("falls back to max_tokens when max_completion_tokens is absent", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 4096,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.max_tokens).toBe(4096)
    })

    it("defaults to 8192 when neither max field is set", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.max_tokens).toBe(8192)
    })

    it("forwards top_p", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        top_p: 0.9,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.top_p).toBe(0.9)
    })

    it("omits top_p when null", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        top_p: null,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.top_p).toBeUndefined()
    })

    it("forwards stream flag", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        stream: true,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.stream).toBe(true)
    })

    it("maps user field to metadata.user_id", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        user: "user-123",
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.metadata).toEqual({ user_id: "user-123" })
    })

    it("omits metadata when user is absent", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.metadata).toBeUndefined()
    })
  })

  describe("system message handling", () => {
    it("hoists system message to system field", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hi" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.system).toBe("You are a helpful assistant.")
      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe("user")
    })

    it("concatenates multiple system messages", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "system", content: "Be concise." },
          { role: "user", content: "Hi" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.system).toBe("You are helpful.\n\nBe concise.")
    })

    it("hoists developer role as system", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "developer", content: "Follow these rules." },
          { role: "user", content: "Hi" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.system).toBe("Follow these rules.")
    })

    it("handles system message with content parts array", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          {
            role: "system",
            content: [
              { type: "text", text: "Rule 1." },
              { type: "text", text: "Rule 2." },
            ],
          },
          { role: "user", content: "Hi" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.system).toBe("Rule 1.\n\nRule 2.")
    })

    it("omits system when no system messages exist", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.system).toBeUndefined()
    })
  })

  describe("user message handling", () => {
    it("translates simple string content", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "What is 2+2?" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.messages[0].role).toBe("user")
      const content = result.messages[0].content as Array<{ type: string }>
      expect(content[0].type).toBe("text")
    })

    it("translates content parts with text", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Describe this image:" },
              {
                type: "image_url",
                image_url: {
                  url: "data:image/png;base64,iVBOR...",
                  detail: "high",
                },
              },
            ],
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      const content = result.messages[0].content as Array<{ type: string }>
      expect(content).toHaveLength(2)
      expect(content[0].type).toBe("text")
      expect(content[1].type).toBe("image")
    })

    it("parses base64 image data URL correctly", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: {
                  url: "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                },
              },
            ],
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      const content = result.messages[0].content as Array<{
        type: string
        source?: { media_type: string; data: string }
      }>
      expect(content[0].type).toBe("image")
      expect(content[0].source?.media_type).toBe("image/jpeg")
      expect(content[0].source?.data).toBe("/9j/4AAQSkZJRg...")
    })

    it("skips non-data-url images gracefully", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: { url: "https://example.com/image.png" },
              },
            ],
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      // Should fall back to empty text block since URL images can't be base64-converted
      const content = result.messages[0].content as Array<{ type: string }>
      expect(content[0].type).toBe("text")
    })
  })

  describe("assistant message handling", () => {
    it("translates assistant text content", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "Hi" },
          { role: "assistant", content: "Hello! How can I help?" },
          { role: "user", content: "What's the weather?" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.messages).toHaveLength(3)
      expect(result.messages[1].role).toBe("assistant")
      const content = result.messages[1].content as Array<{
        type: string
        text: string
      }>
      expect(content[0].text).toBe("Hello! How can I help?")
    })

    it("translates assistant tool_calls", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "Search for news" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_abc123",
                type: "function",
                function: {
                  name: "web_search",
                  arguments: '{"query":"latest news"}',
                },
              },
            ],
          },
          {
            role: "tool",
            tool_call_id: "call_abc123",
            content: "Here are the results...",
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      // Should produce: user, assistant (with tool_use), user (with tool_result)
      expect(result.messages).toHaveLength(3)
      expect(result.messages[1].role).toBe("assistant")
      const assistantContent = result.messages[1].content as Array<{
        type: string
      }>
      expect(assistantContent.some((b) => b.type === "tool_use")).toBe(true)

      expect(result.messages[2].role).toBe("user")
      const userContent = result.messages[2].content as Array<{ type: string }>
      expect(userContent.some((b) => b.type === "tool_result")).toBe(true)
    })

    it("handles assistant with both text and tool_calls", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "Search for news" },
          {
            role: "assistant",
            content: "Let me search for that.",
            tool_calls: [
              {
                id: "call_xyz",
                type: "function",
                function: {
                  name: "search",
                  arguments: '{"q":"news"}',
                },
              },
            ],
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      const assistantContent = result.messages[1].content as Array<{
        type: string
      }>
      const hasText = assistantContent.some((b) => b.type === "text")
      const hasToolUse = assistantContent.some((b) => b.type === "tool_use")
      expect(hasText).toBe(true)
      expect(hasToolUse).toBe(true)
    })
  })

  describe("tool message handling", () => {
    it("translates tool result with string content", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "Search" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "search", arguments: '{"q":"test"}' },
              },
            ],
          },
          {
            role: "tool",
            tool_call_id: "call_1",
            content: "Found 3 results",
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      const userMsg = result.messages[2]
      expect(userMsg.role).toBe("user")
      const content = userMsg.content as Array<{
        type: string
        tool_use_id?: string
        content?: string
      }>
      const toolResult = content.find((b) => b.type === "tool_result")
      expect(toolResult).toBeDefined()
      expect(toolResult!.tool_use_id).toBe("call_1")
      expect(toolResult!.content).toBe("Found 3 results")
    })

    it("handles multiple sequential tool results", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "Do two things" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_a",
                type: "function",
                function: { name: "tool_a", arguments: "{}" },
              },
              {
                id: "call_b",
                type: "function",
                function: { name: "tool_b", arguments: "{}" },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_a", content: "Result A" },
          { role: "tool", tool_call_id: "call_b", content: "Result B" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      // Tool results should be in a user message following the assistant
      const lastUserMsg = result.messages.find(
        (m, i) => i > 0 && m.role === "user",
      )
      expect(lastUserMsg).toBeDefined()
      const content = lastUserMsg!.content as Array<{
        type: string
        tool_use_id?: string
      }>
      const toolResults = content.filter((b) => b.type === "tool_result")
      expect(toolResults).toHaveLength(2)
      expect(toolResults[0].tool_use_id).toBe("call_a")
      expect(toolResults[1].tool_use_id).toBe("call_b")
    })
  })

  describe("message alternation", () => {
    it("merges consecutive user messages", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "First message" },
          { role: "user", content: "Second message" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      // Should merge into a single user message
      expect(result.messages).toHaveLength(1)
      expect(result.messages[0].role).toBe("user")
      const content = result.messages[0].content as Array<{
        type: string
        text: string
      }>
      expect(content.length).toBeGreaterThanOrEqual(2)
    })

    it("ensures first message is always from user", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.messages[0].role).toBe("user")
    })

    it("produces at least one message even with only system content", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "system", content: "Be helpful" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.messages.length).toBeGreaterThanOrEqual(1)
      expect(result.system).toBe("Be helpful")
    })
  })

  describe("tools translation", () => {
    it("translates function tools to anthropic format", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get current weather",
              parameters: {
                type: "object",
                properties: {
                  location: { type: "string" },
                },
                required: ["location"],
              },
            },
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toHaveLength(1)
      expect(result.tools![0].name).toBe("get_weather")
      expect(result.tools![0].description).toBe("Get current weather")
      expect(result.tools![0].input_schema).toEqual({
        type: "object",
        properties: { location: { type: "string" } },
        required: ["location"],
      })
    })

    it("handles multiple tools", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tools: [
          {
            type: "function",
            function: { name: "tool_a", parameters: {} },
          },
          {
            type: "function",
            function: {
              name: "tool_b",
              description: "desc",
              parameters: { type: "object" },
            },
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toHaveLength(2)
      expect(result.tools![0].name).toBe("tool_a")
      expect(result.tools![1].name).toBe("tool_b")
    })

    it("omits tools when array is empty", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tools: [],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toBeUndefined()
    })

    it("omits tools when null", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tools: null,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toBeUndefined()
    })
  })

  describe("tool_choice translation", () => {
    it("maps auto to { type: auto }", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: "auto",
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tool_choice).toEqual({ type: "auto" })
    })

    it("maps required to { type: any }", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: "required",
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tool_choice).toEqual({ type: "any" })
    })

    it("maps none to { type: none }", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: "none",
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tool_choice).toEqual({ type: "none" })
    })

    it("maps function choice to { type: tool, name }", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: { type: "function", function: { name: "get_weather" } },
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tool_choice).toEqual({
        type: "tool",
        name: "get_weather",
      })
    })

    it("omits tool_choice when null", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [{ role: "user", content: "Hi" }],
        tool_choice: null,
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tool_choice).toBeUndefined()
    })
  })

  describe("multi-turn conversations", () => {
    it("handles a full conversation with tool use cycle", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "system", content: "You are a weather assistant." },
          { role: "user", content: "What's the weather in London?" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_weather",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: '{"location":"London"}',
                },
              },
            ],
          },
          {
            role: "tool",
            tool_call_id: "call_weather",
            content: '{"temp": 15, "condition": "cloudy"}',
          },
          {
            role: "assistant",
            content: "It's 15 degrees and cloudy in London.",
          },
          { role: "user", content: "Thanks!" },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.system).toBe("You are a weather assistant.")
      // Messages should alternate user/assistant correctly
      for (let i = 1; i < result.messages.length; i++) {
        expect(result.messages[i].role).not.toBe(result.messages[i - 1].role)
      }
    })

    it("handles multiple tool calls in sequence", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4-20250514",
        messages: [
          { role: "user", content: "Compare weather in London and Paris" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: '{"location":"London"}',
                },
              },
              {
                id: "call_2",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: '{"location":"Paris"}',
                },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_1", content: "15C cloudy" },
          { role: "tool", tool_call_id: "call_2", content: "18C sunny" },
          {
            role: "assistant",
            content: "London is 15C and cloudy, Paris is 18C and sunny.",
          },
        ],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      // Verify alternation
      for (let i = 1; i < result.messages.length; i++) {
        expect(result.messages[i].role).not.toBe(result.messages[i - 1].role)
      }

      // Verify tool_use blocks in assistant message
      const assistantMsg = result.messages[1]
      expect(assistantMsg.role).toBe("assistant")
      const aContent = assistantMsg.content as Array<{ type: string }>
      const toolUseBlocks = aContent.filter((b) => b.type === "tool_use")
      expect(toolUseBlocks).toHaveLength(2)
    })
  })
})

// ===========================================================================
// Response translation: Anthropic -> Completions
// ===========================================================================

describe("translateAnthropicResultToCompletions", () => {
  it("translates a simple text response", () => {
    const response: AnthropicResponse = {
      id: "msg_123",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Hello! How can I help?" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 8 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.id).toBe("msg_123")
    expect(result.object).toBe("chat.completion")
    expect(result.model).toBe("claude-sonnet-4-20250514")
    expect(result.choices).toHaveLength(1)
    expect(result.choices[0].message.role).toBe("assistant")
    expect(result.choices[0].message.content).toBe("Hello! How can I help?")
    expect(result.choices[0].finish_reason).toBe("stop")
  })

  it("translates tool_use to tool_calls", () => {
    const response: AnthropicResponse = {
      id: "msg_456",
      type: "message",
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "toolu_abc",
          name: "get_weather",
          input: { location: "London" },
        },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 20, output_tokens: 15 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.choices[0].finish_reason).toBe("tool_calls")
    expect(result.choices[0].message.tool_calls).toHaveLength(1)
    expect(result.choices[0].message.tool_calls![0]).toEqual({
      id: "toolu_abc",
      type: "function",
      function: {
        name: "get_weather",
        arguments: '{"location":"London"}',
      },
    })
  })

  it("handles text + tool_use combined", () => {
    const response: AnthropicResponse = {
      id: "msg_789",
      type: "message",
      role: "assistant",
      content: [
        { type: "text", text: "Let me check that for you." },
        {
          type: "tool_use",
          id: "toolu_xyz",
          name: "search",
          input: { query: "news" },
        },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 30, output_tokens: 25 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.choices[0].message.content).toBe("Let me check that for you.")
    expect(result.choices[0].message.tool_calls).toHaveLength(1)
    expect(result.choices[0].finish_reason).toBe("tool_calls")
  })

  it("ignores thinking blocks in output", () => {
    const response: AnthropicResponse = {
      id: "msg_think",
      type: "message",
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: "Let me reason about this...",
          signature: "sig@id",
        },
        { type: "text", text: "The answer is 42." },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 20 },
    }

    const result = translateAnthropicResultToCompletions(response)

    // Only text should appear - thinking is not part of completions format
    expect(result.choices[0].message.content).toBe("The answer is 42.")
    expect(result.choices[0].message.tool_calls).toBeUndefined()
  })

  it("maps end_turn to stop", () => {
    const response: AnthropicResponse = {
      id: "msg_1",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Done" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    }

    const result = translateAnthropicResultToCompletions(response)
    expect(result.choices[0].finish_reason).toBe("stop")
  })

  it("maps stop_sequence to stop", () => {
    const response: AnthropicResponse = {
      id: "msg_2",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Partial" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "stop_sequence",
      stop_sequence: "\n",
      usage: { input_tokens: 5, output_tokens: 1 },
    }

    const result = translateAnthropicResultToCompletions(response)
    expect(result.choices[0].finish_reason).toBe("stop")
  })

  it("maps max_tokens to length", () => {
    const response: AnthropicResponse = {
      id: "msg_3",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Truncated output..." }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "max_tokens",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 4096 },
    }

    const result = translateAnthropicResultToCompletions(response)
    expect(result.choices[0].finish_reason).toBe("length")
  })

  it("maps tool_use stop reason to tool_calls", () => {
    const response: AnthropicResponse = {
      id: "msg_4",
      type: "message",
      role: "assistant",
      content: [{ type: "tool_use", id: "t1", name: "fn", input: {} }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 5 },
    }

    const result = translateAnthropicResultToCompletions(response)
    expect(result.choices[0].finish_reason).toBe("tool_calls")
  })

  it("maps usage correctly", () => {
    const response: AnthropicResponse = {
      id: "msg_usage",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Hi" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_input_tokens: 80,
      },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.usage!.prompt_tokens).toBe(180) // 100 + 80
    expect(result.usage!.completion_tokens).toBe(50)
    expect(result.usage!.total_tokens).toBe(230) // 180 + 50
    expect(result.usage!.prompt_tokens_details?.cached_tokens).toBe(80)
  })

  it("handles usage without cache", () => {
    const response: AnthropicResponse = {
      id: "msg_nocache",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Hi" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 50, output_tokens: 10 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.usage!.prompt_tokens).toBe(50)
    expect(result.usage!.completion_tokens).toBe(10)
    expect(result.usage!.total_tokens).toBe(60)
    expect(result.usage!.prompt_tokens_details).toBeUndefined()
  })

  it("preserves copilot_usage", () => {
    const response: AnthropicResponse = {
      id: "msg_aiu",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Done" }],
      copilot_usage: { total_nano_aiu: 12000 },
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 2 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.copilot_usage).toEqual({ total_nano_aiu: 12000 })
  })

  it("sets content to null when only tool_use with no text", () => {
    const response: AnthropicResponse = {
      id: "msg_no_text",
      type: "message",
      role: "assistant",
      content: [{ type: "tool_use", id: "t1", name: "fn", input: { a: 1 } }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 5 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.choices[0].message.content).toBeNull()
  })

  it("handles multiple tool_use blocks", () => {
    const response: AnthropicResponse = {
      id: "msg_multi_tool",
      type: "message",
      role: "assistant",
      content: [
        { type: "tool_use", id: "t1", name: "search", input: { q: "a" } },
        { type: "tool_use", id: "t2", name: "fetch", input: { url: "b" } },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 20 },
    }

    const result = translateAnthropicResultToCompletions(response)

    expect(result.choices[0].message.tool_calls).toHaveLength(2)
    expect(result.choices[0].message.tool_calls![0].id).toBe("t1")
    expect(result.choices[0].message.tool_calls![0].function.name).toBe(
      "search",
    )
    expect(result.choices[0].message.tool_calls![1].id).toBe("t2")
    expect(result.choices[0].message.tool_calls![1].function.name).toBe("fetch")
  })
})

// ===========================================================================
// Structured output: forced tool_use for json_schema enforcement
// ===========================================================================

describe("structured output via forced tool_use", () => {
  describe("request translation", () => {
    it("injects synthetic tool when response_format is json_schema", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4.6",
        messages: [{ role: "user", content: "List colours" }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "colours",
            schema: {
              type: "object",
              properties: {
                colours: { type: "array", items: { type: "string" } },
              },
              required: ["colours"],
            },
            strict: true,
          },
        },
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toBeDefined()
      const synthTool = result.tools!.find(
        (t) => t.name === "structured_response",
      )
      expect(synthTool).toBeDefined()
      expect(synthTool!.input_schema).toEqual({
        type: "object",
        properties: { colours: { type: "array", items: { type: "string" } } },
        required: ["colours"],
      })
      expect(result.tool_choice).toEqual({
        type: "tool",
        name: "structured_response",
      })
    })

    it("appends synthetic tool alongside existing tools", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4.6",
        messages: [{ role: "user", content: "Search and format" }],
        tools: [
          {
            type: "function",
            function: {
              name: "search",
              description: "Search",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "result",
            schema: { type: "object", properties: { ok: { type: "boolean" } } },
          },
        },
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toHaveLength(2)
      expect(result.tools![0].name).toBe("search")
      expect(result.tools![1].name).toBe("structured_response")
      // tool_choice overridden to force synthetic tool
      expect(result.tool_choice).toEqual({
        type: "tool",
        name: "structured_response",
      })
    })

    it("does not inject synthetic tool for json_object format", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4.6",
        messages: [{ role: "user", content: "Return JSON" }],
        response_format: { type: "json_object" },
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toBeUndefined()
      expect(result.tool_choice).toBeUndefined()
    })

    it("does not inject synthetic tool when no response_format", () => {
      const payload: ChatCompletionsPayload = {
        model: "claude-sonnet-4.6",
        messages: [{ role: "user", content: "Hello" }],
      }

      const result = translateCompletionsToMessagesPayload(payload)

      expect(result.tools).toBeUndefined()
      expect(result.tool_choice).toBeUndefined()
    })
  })

  describe("response unwrapping", () => {
    it("unwraps synthetic tool_use as text content", () => {
      const response: AnthropicResponse = {
        id: "msg_struct",
        type: "message",
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "toolu_synth",
            name: "structured_response",
            input: { colours: ["red", "blue", "yellow"] },
          },
        ],
        model: "claude-sonnet-4.6",
        stop_reason: "tool_use",
        stop_sequence: null,
        usage: { input_tokens: 20, output_tokens: 15 },
      }

      const result = translateAnthropicResultToCompletions(
        response,
        "structured_response",
      )

      // Should be text content, not tool_calls
      expect(result.choices[0].message.content).toBe(
        '{"colours":["red","blue","yellow"]}',
      )
      expect(result.choices[0].message.tool_calls).toBeUndefined()
      // finish_reason should be stop, not tool_calls
      expect(result.choices[0].finish_reason).toBe("stop")
    })

    it("passes through real tool_use when forcedToolName is set", () => {
      const response: AnthropicResponse = {
        id: "msg_mixed",
        type: "message",
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "toolu_real",
            name: "search",
            input: { q: "test" },
          },
          {
            type: "tool_use",
            id: "toolu_synth",
            name: "structured_response",
            input: { result: "found" },
          },
        ],
        model: "claude-sonnet-4.6",
        stop_reason: "tool_use",
        stop_sequence: null,
        usage: { input_tokens: 30, output_tokens: 20 },
      }

      const result = translateAnthropicResultToCompletions(
        response,
        "structured_response",
      )

      // Real tool should be in tool_calls
      expect(result.choices[0].message.tool_calls).toHaveLength(1)
      expect(result.choices[0].message.tool_calls![0].function.name).toBe(
        "search",
      )
      // Synthetic tool should be in content
      expect(result.choices[0].message.content).toBe('{"result":"found"}')
    })

    it("treats all tool_use as normal when no forcedToolName", () => {
      const response: AnthropicResponse = {
        id: "msg_normal",
        type: "message",
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "structured_response",
            input: { ok: true },
          },
        ],
        model: "claude-sonnet-4.6",
        stop_reason: "tool_use",
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 5 },
      }

      // No forcedToolName - should treat as normal tool_call
      const result = translateAnthropicResultToCompletions(response)

      expect(result.choices[0].message.tool_calls).toHaveLength(1)
      expect(result.choices[0].message.tool_calls![0].function.name).toBe(
        "structured_response",
      )
      expect(result.choices[0].message.content).toBeNull()
    })
  })
})
