import { describe, expect, test } from "bun:test"
import { readFile } from "node:fs/promises"
import { runInNewContext } from "node:vm"

import { createServer } from "~/server"

const pageUrl = new URL("../pages/index.html", import.meta.url)
const origin = "https://gateway.example"
const legacyPrefix = "copilot-api.usage-viewer.credential."

type Listener = () => unknown

class FixtureElement {
  value = ""
  checked = false
  disabled = false
  innerHTML = ""
  textContent = ""
  style = {}
  listeners = new Map<string, Listener>()

  addEventListener(name: string, callback: Listener) {
    this.listeners.set(name, callback)
  }

  querySelector() {
    return new FixtureElement()
  }
}

async function viewer(
  options: {
    endpoint?: string
    stored?: Map<string, string>
    quota?: Record<string, unknown>
    blockedStorage?: boolean
  } = {},
) {
  const elements = new Map<string, FixtureElement>()
  const element = (id: string) => {
    let value = elements.get(id)
    if (!value) {
      value = new FixtureElement()
      elements.set(id, value)
    }
    return value
  }
  const stored = options.stored ?? new Map<string, string>()
  const requests: Array<{ url: string; init: RequestInit }> = []
  const location = new URL(`${origin}/usage-viewer`)
  if (options.endpoint) location.searchParams.set("endpoint", options.endpoint)
  const storage = {
    get length() {
      return stored.size
    },
    key: (index: number) => [...stored.keys()][index] ?? null,
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => {
      stored.set(key, value)
    },
    removeItem: (key: string) => {
      stored.delete(key)
    },
  }
  const window = {
    location,
    history: { pushState() {} },
    get localStorage() {
      if (options.blockedStorage) throw new Error("Storage unavailable")
      return storage
    },
  }
  const html = await readFile(pageUrl, "utf8")
  const asset = html.match(/<script[^>]+src="([^"]*viewer\.js)"[^>]*>/)?.[1]
  const script =
    asset ?
      await readFile(new URL(asset, pageUrl), "utf8")
    : html.match(/<script>([\s\S]*?)<\/script>/)?.[1]
  if (!script) throw new Error("Viewer script is missing")
  let ready: Listener = () => {}
  runInNewContext(script, {
    document: {
      getElementById: element,
      addEventListener: (name: string, callback: Listener) => {
        if (name === "DOMContentLoaded") ready = callback
      },
    },
    window,
    URL,
    URLSearchParams,
    console: { warn() {}, error() {} },
    fetch: (url: string, init: RequestInit) => {
      requests.push({ url, init })
      return Promise.resolve(
        new URL(url).pathname.endsWith("/usage") ?
          Response.json({
            quota_snapshots: {
              chat: options.quota ?? {
                entitlement: 100,
                remaining: 25,
                percent_remaining: 25,
                unlimited: false,
              },
            },
          })
        : new Response("Missing", { status: 404 }),
      )
    },
  })
  ready()
  const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0))
  await flush()
  const fire = async (id: string, event: string) => {
    const listener = element(id).listeners.get(event)
    if (!listener) throw new Error(`Missing ${event} listener for ${id}`)
    // The fixture invokes the registered production handler, including its guards.
    Reflect.apply(listener, undefined, [{ preventDefault() {} }])
    await flush()
  }
  return { element, fire, requests, stored }
}

describe("usage viewer security", () => {
  test.each(["entitlement", "remaining"])(
    "does not render markup from %s",
    async (field) => {
      const page = await viewer({
        endpoint: `${origin}/usage`,
        quota: {
          entitlement: 100,
          remaining: 25,
          percent_remaining: 25,
          unlimited: false,
          [field]: '<b data-audit="untrusted">external value</b>',
        },
      })
      expect(page.element("content-area").innerHTML).not.toContain(
        "<b data-audit=",
      )
      expect(page.element("content-area").innerHTML).toContain("N/A")
    },
  )

  test("preserves numeric quotas and clamps the progress width", async () => {
    const page = await viewer({
      endpoint: `${origin}/usage`,
      quota: {
        entitlement: 100,
        remaining: 25,
        percent_remaining: -30,
        unlimited: false,
      },
    })
    const rendered = page.element("content-area").innerHTML
    expect(rendered).toContain("75 / 100")
    expect(rendered).toContain("width: 100%")
    expect(rendered).not.toContain("width: 130%")
  })

  test("does not automatically fetch an off-origin endpoint, even on period change", async () => {
    const page = await viewer({ endpoint: "https://other.example/usage" })
    expect(page.requests).toHaveLength(0)
    await page.fire("token-usage-period", "change")
    expect(page.requests).toHaveLength(0)
    await page.fire("endpoint-form", "submit")
    expect(page.requests).toHaveLength(4)
  })

  test("rejects endpoint URLs containing credentials or non-HTTP schemes", async () => {
    for (const endpoint of [
      "https://name:password@other.example/usage",
      "data:text/plain,test",
    ]) {
      const page = await viewer({ endpoint })
      await page.fire("endpoint-form", "submit")
      expect(page.requests).toHaveLength(0)
    }
  })

  test("keeps newly entered credentials in memory by default", async () => {
    const page = await viewer()
    page.element("endpoint-url").value = `${origin}/usage`
    await page.fire("endpoint-url", "change")
    page.element("x-api-key").value = "viewer-fixture-key"
    await page.fire("x-api-key", "input")
    await page.fire("endpoint-form", "submit")
    expect(page.stored.size).toBe(0)
    expect(new Headers(page.requests[0].init.headers).get("x-api-key")).toBe(
      "viewer-fixture-key",
    )
    expect(
      page.requests.every(
        ({ init }) => init.redirect === "error" && init.credentials === "omit",
      ),
    ).toBe(true)
  })

  test("remembers credentials only with explicit consent and removes them when unchecked", async () => {
    const page = await viewer({ endpoint: `${origin}/usage` })
    page.element("x-api-key").value = "viewer-fixture-key"
    await page.fire("x-api-key", "input")
    page.element("remember-credential").checked = true
    await page.fire("remember-credential", "change")
    const reloaded = await viewer({
      endpoint: `${origin}/usage`,
      stored: page.stored,
    })
    expect(reloaded.element("x-api-key").value).toBe("viewer-fixture-key")
    expect(reloaded.element("remember-credential").checked).toBe(true)
    reloaded.element("remember-credential").checked = false
    await reloaded.fire("remember-credential", "change")
    expect(reloaded.stored.size).toBe(0)
    expect(reloaded.element("x-api-key").value).toBe("viewer-fixture-key")
  })

  test("removes legacy saved keys without touching unrelated storage", async () => {
    const page = await viewer({
      stored: new Map([
        [`${legacyPrefix}${origin}`, "old-fixture-key"],
        ["copilot-api.usage-viewer.x-api-key", "legacy-fixture-key"],
        ["other-app", "keep"],
      ]),
    })
    expect([...page.stored.entries()]).toEqual([["other-app", "keep"]])
    expect(page.element("x-api-key").value).toBe("")
  })

  test("does not forward one origin's key when the endpoint changes", async () => {
    const page = await viewer({ endpoint: `${origin}/usage` })
    page.element("x-api-key").value = "viewer-fixture-key"
    await page.fire("x-api-key", "input")
    page.requests.length = 0
    page.element("endpoint-url").value = "https://other.example/usage"
    await page.fire("endpoint-form", "submit")
    expect(page.requests).toHaveLength(4)
    expect(
      page.requests.every(
        ({ init }) => !new Headers(init.headers).has("x-api-key"),
      ),
    ).toBe(true)
  })

  test("continues working when browser storage is unavailable", async () => {
    const page = await viewer({ blockedStorage: true })
    page.element("x-api-key").value = "viewer-fixture-key"
    await page.fire("x-api-key", "input")
    await page.fire("endpoint-form", "submit")
    expect(page.requests).toHaveLength(4)
    expect(new Headers(page.requests[0].init.headers).get("x-api-key")).toBe(
      "viewer-fixture-key",
    )
  })

  test("serves only local assets with viewer-scoped browser protections", async () => {
    const app = createServer({
      networkExposed: true,
      getApiKeys: () => ["server-fixture-key"],
    })
    const response = await app.request(`${origin}/usage-viewer`)
    expect(response.status).toBe(200)
    const html = await response.text()
    expect(html).not.toMatch(/<(?:script|link)[^>]+(?:src|href)="https?:/)
    expect(html).not.toMatch(/<script\s*>/)
    expect(response.headers.get("content-security-policy")).toContain(
      "script-src 'self'",
    )
    expect(response.headers.get("content-security-policy")).toContain(
      "frame-ancestors 'none'",
    )
    expect(response.headers.get("referrer-policy")).toBe("no-referrer")
    const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)]
    expect(assets.length).toBeGreaterThan(0)
    for (const [, asset] of assets) {
      const result = await app.request(
        new URL(asset, `${origin}/usage-viewer`).href,
      )
      expect(result.status).toBe(200)
      expect(result.headers.get("x-content-type-options")).toBe("nosniff")
      expect((await result.text()).length).toBeGreaterThan(0)
    }
    expect((await app.request(`${origin}/usage`)).status).toBe(401)
    expect((await app.request(`${origin}/v1/models`)).status).toBe(401)
    expect(
      (await app.request(`${origin}/usage-viewer-assets/not-public.txt`))
        .status,
    ).not.toBe(200)
    expect(
      (await app.request(`${origin}/`)).headers.has("content-security-policy"),
    ).toBe(false)
  })
})
