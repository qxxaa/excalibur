# Usage viewer security

The usage viewer is available at `/usage-viewer`. Its assets are included in the application under `pages/usage-viewer-assets/`; it does not fetch scripts, styles, icons or fonts from external services. The root Excalibur README remains unchanged.

## Credentials and endpoints

- The viewer accepts a gateway API key or an Authorization Bearer credential. This is separate from the server's GitHub token supplied through `COPILOT_API_GITHUB_TOKEN` or the existing `GH_TOKEN` entrypoint fallback. Server token loading and API authentication are unchanged.
- Credentials stay in memory by default. The **Remember this gateway key** checkbox explicitly permits storage for the selected endpoint origin. Unchecking it removes the saved key without clearing the current in-memory value.
- Previously automatically saved keys are removed on first load of the updated viewer. They require re-entry; old storage is not treated as consent to the new persistence policy. Unrelated browser storage is preserved.
- Remembered keys remain accessible to JavaScript running on this origin. Remembering is a convenience trade-off, not an encrypted credential store.
- A same-origin endpoint supplied in the URL can load automatically. An external endpoint requires an explicit Refresh; changing the period or paging does not bypass this requirement. Only HTTP/HTTPS URLs without embedded usernames or passwords are accepted.
- Fetches omit ambient browser credentials and reject redirects rather than risk forwarding a gateway key to a different destination. Reverse-proxy arrangements requiring cookies for the API need a separate compatibility decision; header-based gateway authentication remains supported.

## Browser policy

The page carries a Content Security Policy in both a meta element and its gateway response. Scripts are restricted to this origin; inline scripts and dynamic evaluation are not allowed. The gateway also prohibits framing and sends `Referrer-Policy: no-referrer` and `X-Content-Type-Options: nosniff`.

Inline styles remain permitted for the existing chart and progress-bar geometry. Quota values are accepted only as finite numbers and progress widths are clamped to 0-100. Unknown values display as `N/A` rather than becoming HTML. HTTP/HTTPS connections remain available for explicitly selected external gateways; CSP is not a network allowlist.

Only the four named viewer asset paths are public. API and admin authentication are not relaxed, and request input is not used to select filesystem files. Normal application builds, Docker packaging and the static Pages deployment include the existing `pages/` directory; no asset download is needed at runtime.

## Local asset maintenance

`utilities.css` is generated using Tailwind CSS 3.4.17. `icons.js` bundles only the six Lucide 0.378.0 icons used by the page and the icon renderer. The build inputs and third-party licence notices are committed alongside them. The viewer uses system fonts.

From the repository root, the assets can be regenerated with Node/npm and Bun installed:

```sh
npm install --prefix /tmp/excalibur-viewer-assets --ignore-scripts --package-lock=false --no-audit --no-fund tailwindcss@3.4.17 lucide@0.378.0
cp pages/usage-viewer-assets/icons-entry.mjs /tmp/excalibur-viewer-assets/icons-entry.mjs
bun build /tmp/excalibur-viewer-assets/icons-entry.mjs --target=browser --format=iife --minify --outfile=pages/usage-viewer-assets/icons.js
/tmp/excalibur-viewer-assets/node_modules/.bin/tailwindcss --input pages/usage-viewer-assets/utilities.input.css --content 'pages/index.html,pages/usage-viewer-assets/viewer.js' --output pages/usage-viewer-assets/utilities.css --minify
```

The two asset packages are pinned; this maintenance command is not a fully locked build-tool dependency graph. Generated assets are reviewed and committed, not rebuilt from the registry during application startup or deployment. Rebuild utilities after adding or removing Tailwind classes, and review generated changes before adoption.

## Verification

```sh
bun test tests/usage-viewer-security.test.ts tests/usage-viewer.test.ts tests/request-auth.test.ts tests/server-security.test.ts
bun run typecheck
bun run lint
bun test
bun run build
```

The security suite executes the actual viewer script with controlled DOM, storage and HTTP boundaries. Browser verification additionally checks script execution policy, local assets, icons, rendering, credential reload behaviour, redirects and endpoint admission. No real provider token is required.
