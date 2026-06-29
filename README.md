# ⚔️ Excalibur

*A blade forged to cut through the babel of tongues.*

Excalibur is a proxy that receiveth any request - be it `/v1/messages`, `/v1/chat/completions`, or `/v1/responses` - discerneth the transport favoured by the target model, and translateth between them without counsel from the caller.

Send in any tongue. The blade shall find its mark.

### Capabilities

- Automatic detection of model transport from upstream capabilities
- `/v1/messages` to `/v1/responses` translation
- `/v1/chat/completions` to `/v1/responses` translation
- Strict JSON schema enforcement (Pydantic v2 compatible)
- Reasoning effort passthrough across all transports
- Streaming support for every path

### Quick Start

```bash
docker run -d ghcr.io/qxxaa/excalibur:latest
```

### Lineage

Forged from the work of [caozhiyuan/copilot-api](https://github.com/caozhiyuan/copilot-api), tempered for those whose clients speak only one dialect yet must reach models that answer in another.
