# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root
- **`docs/adr/`**: read ADRs that touch the area you're about to work in

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   │   ├── 0001-dual-database-isolation.md
│   │   └── 0002-mcp-gateway-as-sole-egress.md
│   └── 01-系统总体方案.md ...
├── packages/
│   ├── mcp-gateway/
│   ├── security-plugin/
│   ├── business-api/
│   └── shared-contracts/
├── apps/web/
└── firmware/stm32-sensor-node/
```

**本仓库是单上下文（single-context）**：整个系统共享一套领域词汇（见 `CONTEXT.md`），系统级决策统一放在 `docs/adr/`。虽然目录上是多包布局，但各包不构成独立的限界上下文——它们服务于同一个"工业预测性运维"领域，因此不拆分 `CONTEXT-MAP.md`。

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

本仓库有几个**必须统一**的措辞（详见 `CONTEXT.md` 的"禁用同义词"列），最容易写错的是：

- 一律说 **"MCP 安全网关"** 或 **"网关"**，不要写成 "API 网关"、"代理层"、"中间件层"。
- 一律说 **"安全管控插件层"**，不要写成 "安全插件"、"权限插件"、"拦截器"。
- 一律说 **"business-db"** 与 **"ai-db"**，不要写成 "业务库/主库/MySQL 库"、"向量库/知识库"（后者易与"知识库"这一业务概念混淆）。
- 一律说 **"遥测（telemetry）"** 指 STM32 上报的传感器数据，不要写成 "监控数据"、"采集数据"。

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0002 (MCP 网关作为 Agent 唯一出口), but worth reopening because…_

本仓库最容易被违反的决策是 **ADR-0002**（Agent 唯一出口）与 **ADR-0001**（双库隔离）。任何让 Agent 直连数据库、直连 MQTT、或在网关之外新增数据通路的改动，都必须在输出中显式标注冲突，不允许静默绕过。
