# ⚔️ Excalibur

*A blade forged to cut through the babel of tongues.*

Excalibur is a translation proxy for LLM API transports. It accepts requests in any of the three major formats and delivers them in whichever format the target model requires.

**Accepts:** `/v1/chat/completions` - `/v1/responses` - `/v1/messages`

**Delivers:** `/v1/chat/completions` - `/v1/responses` - `/v1/messages`

Every combination works. Send in any tongue - the blade shall find its mark.

### Capabilities

- Automatic transport detection from upstream model capabilities
- Full translation between Completions, Responses, and Messages
- Structured JSON schema enforcement via forced `tool_use` on Messages-bound paths
- Reasoning effort normalisation across all transports
- Streaming support for every path

### Quick Start

```bash
docker run -d ghcr.io/qxxaa/excalibur:latest
```

### Lineage

Forged from the work of [caozhiyuan/copilot-api](https://github.com/caozhiyuan/copilot-api), tempered for those whose clients speak only one dialect yet must reach models that answer in another.
