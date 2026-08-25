/**
 * Tests for responses-completions-translation.ts
 * (Responses API -> Chat Completions API translation)
 */

import { describe, expect, it } from "bun:test"

import type { ChatCompletionResponse } from "~/lib/types/chat-completions"
import type { ResponsesPayload } from "~/lib/types/responses"

import {
  translateResponsesToCompletionsPayload,
  translateCompletionsResultToResponses,
} from "~/routes/responses/responses-completions-translation"

// ===========================================================================
// Payload: Responses -> Completions
// ===========================================================================

describe("translateResponsesToCompletionsPayload", () => {
  it("translates simple string input to user message", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: "Hello world",
      stream: false,
      store: false,
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.model).toBe("gemini-3.5-flash")
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].role).toBe("user")
    expect(result.messages[0].content).toBe("Hello world")
  })

  it("hoists instructions to system message", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      instructions: "You are a helpful assistant.",
      input: "Hi",
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.messages).toHaveLength(2)
    expect(result.messages[0].role).toBe("system")
    expect(result.messages[0].content).toBe("You are a helpful assistant.")
    expect(result.messages[1].role).toBe("user")
    expect(result.messages[1].content).toBe("Hi")
  })

  it("translates structured input items", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: [
        { role: "user", content: "What is 2+2?" },
        { role: "assistant", content: "Four" },
        { role: "user", content: "And 3+3?" },
      ],
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.messages).toHaveLength(3)
    expect(result.messages[0]).toEqual({
      role: "user",
      content: "What is 2+2?",
    })
    expect(result.messages[1]).toEqual({ role: "assistant", content: "Four" })
    expect(result.messages[2]).toEqual({ role: "user", content: "And 3+3?" })
  })

  it("translates function_call and function_call_output items", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: [
        { role: "user", content: "What is the weather?" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "get_weather",
          arguments: '{"location":"London"}',
          status: "completed",
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "Sunny, 22C",
        },
      ],
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.messages).toHaveLength(3)
    expect(result.messages[1].role).toBe("assistant")
    expect(result.messages[1].tool_calls).toHaveLength(1)
    expect(result.messages[1].tool_calls![0].id).toBe("call_1")
    expect(result.messages[1].tool_calls![0].function.name).toBe("get_weather")
    expect(result.messages[2].role).toBe("tool")
    expect(result.messages[2].content).toBe("Sunny, 22C")
  })

  it("translates function tools", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: "Test",
      tools: [
        {
          type: "function",
          name: "search",
          description: "Search the web",
          parameters: {
            type: "object",
            properties: { q: { type: "string" } },
          },
          strict: null,
        },
      ],
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].type).toBe("function")
    expect(result.tools![0].function.name).toBe("search")
    expect(result.tools![0].function.description).toBe("Search the web")
  })

  it("skips non-function tools", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: "Test",
      tools: [
        { type: "web_search" } as Record<string, unknown>,
        {
          type: "function",
          name: "search",
          parameters: null,
          strict: null,
        },
      ],
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.tools).toHaveLength(1)
    expect(result.tools![0].function.name).toBe("search")
  })

  it("translates tool_choice string values", () => {
    for (const choice of ["none", "auto", "required"] as const) {
      const payload: ResponsesPayload = {
        model: "gemini-3.5-flash",
        input: "Test",
        tool_choice: choice,
      }
      const result = translateResponsesToCompletionsPayload(payload)
      expect(result.tool_choice).toBe(choice)
    }
  })

  it("translates tool_choice function object", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: "Test",
      tool_choice: { type: "function", name: "search" },
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.tool_choice).toEqual({
      type: "function",
      function: { name: "search" },
    })
  })

  it("maps reasoning effort", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: "Test",
      reasoning: { effort: "medium" },
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.reasoning_effort).toBe("medium")
  })

  it("maps max_output_tokens to max_tokens", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: "Test",
      max_output_tokens: 1000,
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.max_tokens).toBe(1000)
  })

  // Structured output: text.format.json_schema -> response_format.json_schema
  describe("structured output", () => {
    it("translates text.format.json_schema to response_format", () => {
      const payload = {
        model: "gemini-3.5-flash",
        input: "List colours",
        text: {
          format: {
            type: "json_schema",
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
      } as ResponsesPayload

      const result = translateResponsesToCompletionsPayload(payload)

      expect(result.response_format).toEqual({
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
      })
    })

    it("translates text.format.json_object to response_format", () => {
      const payload = {
        model: "gemini-3.5-flash",
        input: "Return JSON",
        text: { format: { type: "json_object" } },
      } as ResponsesPayload

      const result = translateResponsesToCompletionsPayload(payload)

      expect(result.response_format).toEqual({ type: "json_object" })
    })

    it("does not set response_format when no text.format", () => {
      const payload: ResponsesPayload = {
        model: "gemini-3.5-flash",
        input: "Hello",
      }

      const result = translateResponsesToCompletionsPayload(payload)

      expect(result.response_format).toBeUndefined()
    })
  })

  it("translates multi-part content", () => {
    const payload: ResponsesPayload = {
      model: "gemini-3.5-flash",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "Describe this image" },
            {
              type: "input_image",
              image_url: "data:image/png;base64,abc123",
              detail: "high",
            },
          ],
        },
      ],
    }

    const result = translateResponsesToCompletionsPayload(payload)

    expect(result.messages).toHaveLength(1)
    const content = result.messages[0].content as unknown as Array<
      Record<string, unknown>
    >
    expect(content).toHaveLength(2)
    expect(content[0]).toEqual({ type: "text", text: "Describe this image" })
    expect(content[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,abc123", detail: "high" },
    })
  })
})

// ===========================================================================
// Result: Completions -> Responses
// ===========================================================================

describe("translateCompletionsResultToResponses", () => {
  it("translates text response", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl-123",
      object: "chat.completion",
      created: 1234567890,
      model: "gemini-3.5-flash",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hello!" },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    }

    const result = translateCompletionsResultToResponses(response)

    expect(result.id).toBe("cmpl-123")
    expect(result.object).toBe("response")
    expect(result.model).toBe("gemini-3.5-flash")
    expect(result.output_text).toBe("Hello!")
    expect(result.status).toBe("completed")
    expect(result.output).toHaveLength(1)
    expect(result.output[0].type).toBe("message")
    expect(result.usage?.input_tokens).toBe(10)
    expect(result.usage?.output_tokens).toBe(5)
  })

  it("translates tool calls", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl-456",
      object: "chat.completion",
      created: 1234567890,
      model: "gemini-3.5-flash",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "tc_1",
                type: "function",
                function: {
                  name: "search",
                  arguments: '{"q":"test"}',
                },
              },
            ],
          },
          logprobs: null,
          finish_reason: "tool_calls",
        },
      ],
    }

    const result = translateCompletionsResultToResponses(response)

    expect(result.output).toHaveLength(1)
    expect(result.output[0].type).toBe("function_call")
    const fc = result.output[0] as {
      name: string
      arguments: string
      call_id: string
    }
    expect(fc.name).toBe("search")
    expect(fc.arguments).toBe('{"q":"test"}')
    expect(fc.call_id).toBe("tc_1")
  })

  it("maps length finish_reason to incomplete status", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl-789",
      object: "chat.completion",
      created: 1234567890,
      model: "gemini-3.5-flash",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Truncated..." },
          logprobs: null,
          finish_reason: "length",
        },
      ],
    }

    const result = translateCompletionsResultToResponses(response)

    expect(result.status).toBe("incomplete")
    expect(result.incomplete_details?.reason).toBe("max_output_tokens")
  })

  it("preserves copilot_usage", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl-aiu",
      object: "chat.completion",
      created: 1234567890,
      model: "gemini-3.5-flash",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "OK" },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      copilot_usage: { total_nano_aiu: 5000 },
    }

    const result = translateCompletionsResultToResponses(response)

    expect(result.copilot_usage).toEqual({ total_nano_aiu: 5000 })
  })
})
