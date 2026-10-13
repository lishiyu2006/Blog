# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

本项目暂无远端仓库，问题与规格全部以本地 Markdown 文件形式追踪。

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each issue file
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.

## 本项目建议的 feature-slug 命名

按系统分层命名，与 `docs/` 编号对齐，便于答辩时回溯：

| feature-slug | 对应范围 |
| --- | --- |
| `security-plugin-layer` | 安全管控插件层（**主实现部分**） |
| `mcp-gateway` | MCP 安全网关服务 |
| `agent-core` | OpenClaw 二次开发、ReAct 循环、Skill 插件 |
| `dual-db` | 双库分离与权限边界 |
| `web-console` | Vue3-TS 管理界面 |
| `stm32-node` | STM32 边缘终端与 MQTT 链路 |
| `proto-validation` | 端到端原型验证与演示脚本 |

## 为什么不用远端 tracker

- 毕设单人开发，无协作合并需求，本地文件省去账号与网络依赖。
- 答辩材料需要"可离线打开的证据"，Markdown 文件可直接截图或导出 PDF。
- 若后续要迁到 GitHub/GitLab，重跑 `setup-matt-pocock-skills` 并改选 tracker 即可，本文件会被覆盖。
