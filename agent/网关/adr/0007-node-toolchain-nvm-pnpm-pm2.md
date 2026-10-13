# ADR-0007：Node 工具链采用 NVM → PNPM → PM2

- **状态**：已采纳
- **日期**：2026-09-15
- **决策者**：毕设作者
- **相关文档**：`docs/08-部署与运维.md`、`docs/adr/0003-openclaw-as-agent-core.md`

## 背景

本系统有多个 Node.js 进程需要同时运行与守护：

- MCP 安全网关（TS）
- 业务接口服务（TS）
- OpenClaw Gateway（含安全管控插件层）
- Web 前端构建产物（静态托管，可选由 Nginx 或 Node 服务承载）

这些进程对 Node 版本的要求**并不一致**——尤其是 OpenClaw 有明确的版本下限。用系统全局 Node 会导致"升级一个、崩一个"。

## 决策

采用三段式工具链，职责单一、逐层向上：

```
NVM  ──►  PNPM  ──►  PM2
(管理 Node 版本)  (安装依赖)  (守护 Node 进程)
```

1. **NVM 管理 Node 版本**：在仓库根放 `.nvmrc` 固定版本；每个服务通过 `nvm use` 切到该版本再启动。
2. **PNPM 安装依赖**：仓库用 pnpm workspace 管理多包（`packages/*`、`apps/*`）；只用 `pnpm-lock.yaml`，不引入 npm/yarn 的 lockfile。
3. **PM2 守护 Node 进程**：统一通过 `deploy/ecosystem.config.cjs` 声明所有进程（名称、脚本、实例数、重启策略、日志路径、环境变量文件），不使用手工 `node xxx.js` 起生产进程。

## 关键版本约束（必须核对，写死在 `.nvmrc` 与文档里）

OpenClaw 官方要求：**Node 24.15+（推荐）**，或 **Node 22 LTS ≥ 22.22.3**，或 **Node 25.9+**。

对本机环境的直接影响：

| 本机可用 Node | 版本 | 是否满足 OpenClaw |
| --- | --- | --- |
| 托管版本 | 22.22.2 | ❌ 低于 22.22.3 下限 |
| 系统版本 | 22.23.1 | ✅ 满足 22 LTS 分支 |

→ **结论**：默认用 NVM 安装并固定 **Node 24.x（≥24.15）**，这是官方推荐分支，也避免卡在 22 LTS 的下限边缘。22.23.1 可作为回退。

## 理由

1. **版本冲突是真实的，不是假想**。本机已存在两个 22.x 版本且其中一个不满足 OpenClaw 下限——这正是 NVM 存在的理由。
2. **PNPM 的 workspace 与硬链接机制**适合本仓库的 monorepo 布局（`shared-contracts` 被多个包引用），且安装快、磁盘占用低。
3. **PM2 提供了答辩需要的可观测性**：`pm2 list` 一屏展示所有进程状态、重启次数、CPU/内存；`pm2 logs` 统一收集日志；`pm2 save` + `pm2 startup` 支持开机自启。这些都是现场演示的加分项。
4. **配置即代码**。`ecosystem.config.cjs` 入库，进程拓扑可被评审，而不是靠口头描述。

## 备选方案与拒绝理由

| 方案 | 为什么没选 |
| --- | --- |
| 不用 NVM，直接用系统 Node | 多服务版本需求不同时必然冲突；本机已存在不满足 OpenClaw 下限的版本，风险已现。 |
| 用 npm 而非 PNPM | monorepo 下 npm workspace 的依赖提升与磁盘占用更差；PNPM 的严格依赖树也能更早暴露"未声明依赖"。 |
| 用 Docker Compose 编排全部服务 | 隔离性最好，但演示环境依赖 Docker 可用性，现场风险高；且固件侧仍需本机工具链。可作为落地期的演进方向记录在 `docs/08-部署与运维.md`。 |
| 用 systemd / Windows 服务直接托管 | 跨平台一致性差（开发在 Windows、部署可能在 Linux），PM2 两者都能用；且 PM2 的日志聚合对答辩演示更友好。 |
| 用 `nodemon` / `tsx watch` 做"守护" | 那是开发热重载工具，不是进程守护；崩溃后不会拉起，也不提供开机自启。 |

## 后果

**正面**

- 版本、依赖、进程三层各自有唯一权威配置（`.nvmrc`、`pnpm-lock.yaml`、`ecosystem.config.cjs`），环境可复现。
- `pm2 list` / `pm2 logs` / `pm2 monit` 可直接用于答辩现场演示系统运行状态。
- 新增服务只需在 ecosystem 配置中加一段，不改变工具链。

**负面 / 代价**

- PM2 守护的是**进程**，不解决进程内的内存泄漏与连接池耗尽；需要在 `docs/08-部署与运维.md` 中补充健康检查与内存阈值重启策略（`max_memory_restart`）。
- Windows 下 PM2 的开机自启依赖 `pm2-windows-startup` 或任务计划程序，与 Linux 的 `pm2 startup` 不同，需要分别说明。
- 环境变量通过 PM2 的 `env_file` / `env` 注入，意味着**每个进程的环境变量必须严格区分**——这正是本课题要利用的一点：Agent 进程的环境变量里绝不能出现 `business-db` 连接串。

## 由本决策直接推导出的安全约束

`deploy/ecosystem.config.cjs` 中，各进程的环境变量必须**按最小权限分别配置**：

| 进程 | 允许持有的凭证 | 明确不得持有 |
| --- | --- | --- |
| `mcp-gateway` | `BUSINESS_DB_URL`、`AI_DB_URL`、`MQTT_URL`、`MQTT_USER/PASS`、JWT 公钥 | — |
| `business-api` | `BUSINESS_DB_URL`、JWT 私钥 | `AI_DB_URL`、MQTT 凭证 |
| `openclaw`（含安全管控插件层） | `GATEWAY_MCP_ENDPOINT`、`AGENT_ID`、网关签发的 Agent 令牌 | `BUSINESS_DB_URL`、`AI_DB_URL`、`MQTT_*` **全部不得出现** |

这张表是 `docs/09-原型验证方案.md` 中"凭证隔离验证"用例的判定依据。
