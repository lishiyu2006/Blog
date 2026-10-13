# 04 · MCP 安全网关设计

> **文档定位**：定义系统的**安全边界组件**——MCP 安全网关的接口形态、身份体系、五个 MCP 工具的完整契约、字段裁剪规则、审计落库语义、MQTT 与向量库的接入方式，以及错误码与降级策略。
>
> **关联文档**：`docs/01-系统总体方案.md`、`docs/02-安全管控插件层设计.md`、`docs/05-数据库设计.md`、`docs/07-STM32边缘终端设计.md`、`docs/adr/0002-mcp-gateway-as-sole-egress.md`、`docs/adr/0004-security-plugin-layer-in-process.md`

---

## 1. 定位

MCP 安全网关是本系统**唯一的数据出口**，也是**真正的安全边界**。

它同时承担四个角色：

| 角色 | 内容 |
| --- | --- |
| **统一入口** | 接收 Agent 调用（MCP 工具调用）与人类用户调用（Web 前端经业务侧 JWT 的查询请求），是系统对外的唯一数据通路 |
| **身份解析** | 解析 Agent 令牌 / 工程师 JWT，得到调用方身份与上下文 |
| **策略与裁剪** | 判定放行/拒绝，对放行请求的结果做字段白名单裁剪，只回传最小数据集 |
| **审计权威落点** | 每一次调用（含被拒）写入 `business-db.audit_log`；写入失败即调用失败（fail-closed） |

**它是唯一持有 `business-db` / `ai-db` / MQTT 凭证的 Agent 侧组件。** Agent 与安全管控插件层都不持有这些凭证。

---

## 2. 技术栈与进程形态

| 项 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Node.js + TypeScript | 与 OpenClaw 同语言，可共享契约类型 |
| Node 版本 | 24.15+（见 ADR-0007） | 与 OpenClaw 保持一致 |
| HTTP 框架 | Fastify 或 Express | 原型期任选，倾向 Fastify（schema 校验与性能） |
| MCP 传输 | HTTP（`/mcp`）为主，stdio 为辅 | HTTP 便于 PM2 守护与独立健康检查 |
| 数据库客户端 | better-sqlite3（原型）/ pg（落地） | 同步 API 简化原型事务 |
| MQTT 客户端 | mqtt.js | 唯一订阅方 |
| 向量库客户端 | Chroma HTTP 客户端 或 Qdrant JS SDK | 只读 |
| 校验 | zod / TypeBox | 入参 schema 与共享契约类型同源 |
| 进程守护 | PM2（`mcp-gateway`） | 见 `docs/08-部署与运维.md` |

**监听地址**：`127.0.0.1:7801`（仅本机）。不暴露 LAN。

---

## 3. 目录结构

```
packages/mcp-gateway/
├── package.json
├── tsconfig.json
└── src/
    ├── index.ts                # 进程入口：装配 Fastify + MCP + MQTT
    ├── config.ts               # 环境变量读取与校验（缺失即启动失败）
    ├── auth/
    │   ├── agent-token.ts      # Agent 令牌签发与校验（HMAC / JWT）
    │   ├── user-jwt.ts         # 工程师 JWT 校验
    │   └── context.ts          # 统一构造 RequestContext
    ├── policy/
    │   ├── engine.ts           # 网关侧策略判定（独立于插件层）
    │   └── rules.ts            # 网关侧规则（不信任 L1 判定）
    ├── tools/                  # 每个 MCP 工具一个文件
    │   ├── index.ts            # 工具注册表（契约 + handler）
    │   ├── mqtt.read_sensor.ts
    │   ├── kb.search.ts
    │   ├── db.query_device.ts
    │   ├── db.query_workorder.ts
    │   └── report.write.ts
    ├── redact/
    │   └── whitelist.ts        # 字段白名单裁剪（工具级配置）
    ├── audit/
    │   ├── writer.ts           # 审计落库（fail-closed）
    │   └── schema.ts
    ├── db/
    │   ├── business.ts         # business-db 连接（唯一持有者）
    │   └── ai.ts               # ai-db 只读连接（唯一持有者）
    ├── mqtt/
    │   ├── subscriber.ts       # 订阅 iot/+/telemetry
    │   ├── ingest.ts           # 校验 + 批量写入 sensor_data
    │   └── publisher.ts        # 受限发布（下行指令，落地期）
    ├── routes/
    │   ├── health.ts           # GET /health
    │   ├── tools-invoke.ts     # POST /tools/invoke（MCP 工具调用）
    │   └── audit-report.ts     # POST /audit/report（插件层上报审计）
    └── types.ts
```

---

## 4. 接口形态

### 4.1 对外接口清单

| 方法 | 路径 | 调用方 | 鉴权 | 用途 |
| --- | --- | --- | --- | --- |
| `POST` | `/mcp` | Agent（MCP 客户端） | Agent 令牌 | 标准 MCP 工具调用 |
| `POST` | `/tools/invoke` | 安全管控插件层 | Agent 令牌 | 简化的工具调用（供插件层转发） |
| `POST` | `/audit/report` | 安全管控插件层 | Agent 令牌 | 上报 L1 的判定审计 |
| `GET` | `/tools/list` | Agent / 插件层 | Agent 令牌 | 工具发现（返回封闭清单） |
| `GET` | `/health` | PM2 / 运维 | 无（仅本机） | 健康检查 |
| `GET` | `/metrics` | 运维 | 无（仅本机） | 轻量指标（可选） |

### 4.2 `/tools/invoke` 请求/响应

**请求**

```http
POST /tools/invoke HTTP/1.1
Host: 127.0.0.1:7801
Authorization: Bearer <AGENT_TOKEN>
x-agent-id: ops-analyst
x-trace-id: 7f3c9a12-...
Content-Type: application/json

{
  "tool": "mqtt.read_sensor",
  "params": {
    "deviceId": "DEV-003",
    "from": "2026-09-15T13:00:00+08:00",
    "to":   "2026-09-15T14:00:00+08:00",
    "limit": 300
  }
}
```

**成功响应**

```json
{
  "ok": true,
  "traceId": "7f3c9a12-...",
  "tool": "mqtt.read_sensor",
  "data": [
    { "temp": 63.2, "vibration": 4.81, "time": "2026-09-15T13:00:00+08:00" },
    { "temp": 63.5, "vibration": 4.92, "time": "2026-09-15T13:01:00+08:00" }
  ],
  "meta": {
    "rowCount": 60,
    "returnedFields": ["temp", "vibration", "time"],
    "redactedFields": ["id", "quality", "trace_id", "received_at", "unit"],
    "durationMs": 37
  }
}
```

**被拒响应**

```json
{
  "ok": false,
  "traceId": "7f3c9a12-...",
  "error": {
    "code": "DEVICE_OUT_OF_SCOPE",
    "message": "设备 DEV-009 不在本次调用的授权范围内",
    "hint": "授权设备：DEV-003"
  }
}
```

`meta.redactedFields` 的存在很重要：它让"裁剪确实发生了"这件事**可被观测、可被截图**，是答辩时的直接证据。

---

## 5. 身份体系

### 5.1 两类调用方

| 调用方 | 凭证 | 身份字段 | 授权范围来源 |
| --- | --- | --- | --- |
| Agent | Agent 令牌（网关签发，短时效） | `agentId`、`sessionId`、`taskId` | 任务记录中声明的 `authorizedDevices` |
| 工程师 | 业务侧 JWT | `userId`、`roles` | RBAC 角色权限（`docs/05`） |

### 5.2 Agent 令牌

- **签发**：Agent 会话启动时，由 `business-api` 携带任务 ID 向网关申请；网关校验任务合法性后签发。
- **载荷**：`{ agentId, sessionId, taskId, authorizedDevices, exp }`
- **时效**：短时效（原型期 30 分钟），与会话生命周期绑定。
- **存储**：仅存于 OpenClaw 进程环境变量（PM2 注入），**不落盘到配置文件**。
- **校验**：HMAC-SHA256 签名；校验失败 ⇒ `401`，且记审计。

**关键设计**：`authorizedDevices` 写进令牌，意味着**授权范围由网关签发时决定，Agent 无法自行扩大**。这与 `docs/02` 中插件层 R3 规则形成双保险：插件层从令牌解析出范围做第一次判定，网关从同一令牌做第二次判定。

### 5.3 统一请求上下文

```ts
// packages/mcp-gateway/src/auth/context.ts
export interface RequestContext {
  traceId: string;
  actorType: 'agent' | 'engineer';
  actorId: string;
  sessionId?: string;
  taskId?: string;
  authorizedDevices: Set<string>;
  roles: string[];           // 工程师的 RBAC 角色
  startedAt: number;
}
```

所有策略规则、字段裁剪、审计记录都以 `RequestContext` 为唯一输入，避免"某个工具偷偷用了别处的身份信息"。

---

## 6. 网关侧策略校验

网关**不信任插件层的判定结果**，独立复核（见 ADR-0004、`docs/02` §5）。

| # | 网关规则 | 说明 | 失败码 |
| --- | --- | --- | --- |
| G1 | 令牌有效性与时效 | 签名、`exp`、`sessionId` 一致性 | `UNAUTHORIZED` |
| G2 | 工具在白名单内 | 工具的 `enabled` 与 `actorType` 匹配 | `TOOL_NOT_ALLOWED` |
| G3 | 入参 schema 校验 | 与 `shared-contracts` 同源 | `PARAM_INVALID` |
| G4 | 设备授权范围 | `deviceId ∈ authorizedDevices` | `DEVICE_OUT_OF_SCOPE` |
| G5 | 时间窗上限 | 超限 ⇒ **收窄并记录**（不拒绝，避免打断 Agent 流程） | — |
| G6 | 条数上限 | 超限 ⇒ 截断并记录 | — |
| G7 | 写类工具的角色要求 | `report.write` 需 `taskId` 非空 | `FORBIDDEN` |
| G8 | 速率限制 | 令牌桶，按 `agentId` 与全局双维度 | `RATE_LIMITED` |
| G9 | 审计可用性 | 见 §8 | `AUDIT_UNAVAILABLE` |

> **G5 与插件层 R4 的区别**：插件层对超窗时间直接 REWRITE 并告知模型；网关对超窗做"收窄 + 静默记录"。前者是**给模型的反馈**，后者是**最后一道防线**。两层都做，语义不冲突。

---

## 7. 五个 MCP 工具的完整契约

### 7.1 工具总表

| 工具 | 读/写 | 数据源 | 返回字段白名单 | 授权要求 |
| --- | --- | --- | --- | --- |
| `mqtt.read_sensor` | 读 | `business-db.sensor_data` | `temp`、`vibration`、`time` | 设备在授权范围 |
| `kb.search` | 读 | `ai-db` 向量集合 | `chunk`、`source`、`score` | 任意 Agent |
| `db.query_device` | 读 | `business-db.device` | `model`、`location`、`commissionedAt` | 设备在授权范围 |
| `db.query_workorder` | 读 | `business-db.work_order` | `orderNo`、`status`、`handledAt` | 设备在授权范围 |
| `report.write` | 写 | `business-db.agent_report` | 返回 `reportId` | 需 `taskId` |

**没有任何工具能执行任意 SQL，也没有任何工具能发布 MQTT 消息。**

### 7.2 `mqtt.read_sensor`

| 项 | 内容 |
| --- | --- |
| 用途 | 读取指定设备在时间窗内的遥测数据 |
| 入参 | `deviceId: string`（必填）、`from: ISO8601`（必填）、`to: ISO8601`（必填）、`limit?: number`（默认 300，上限 500） |
| 出参 | `Array<{ temp: number; vibration: number; time: string }>` |
| 白名单外字段 | `id`、`quality`、`trace_id`、`received_at`、`unit`（**裁剪并记入 `meta.redactedFields`**） |
| 降采样 | 当行数 > `limit` 时，按等间隔采样并**强制保留窗口内的最大值与最小值点** |
| 错误码 | `PARAM_INVALID`、`DEVICE_OUT_OF_SCOPE`、`RATE_LIMITED` |
| 数据源选择 | 优先读 `business-db.sensor_data`（权威来源）；若时间窗超出已落库范围，可选从网关内存缓存补齐（原型期不做） |
| **存储形态与返回形态** | `sensor_data` 是**窄表**（`device_id` / `metric` / `value` / `unit` / `ts`，一行一指标）。本工具在返回前做**透视（pivot）与重命名**，输出宽表 `{temp, vibration, time}`。这个转换正是"网关是加工层而非透明代理"的体现：数据库的存储形状与 Agent 看到的契约形状解耦，Agent 无法通过构造参数去查其他 `metric` 或其他表 |

**为什么用数据库而不是直连 MQTT 实时订阅**：见 ADR-0005。Agent 的取数应是"查询语义"，而不是"订阅语义"。

### 7.3 `kb.search`

| 项 | 内容 |
| --- | --- |
| 用途 | 在故障知识库中做向量检索 |
| 入参 | `query: string`（必填，长度 4-512）、`topK?: number`（默认 5，上限 10）、`deviceModel?: string`（可选，用于元数据过滤）、`minScore?: number`（默认 0.5） |
| 出参 | `Array<{ chunk: string; source: string; score: number }>` |
| 出参约束 | `chunk` 单条上限 2000 字符（超长截断并在 `source` 后标注 `[截断]`） |
| 元数据过滤 | `deviceModel` 存在时，限定 `metadata.deviceModel == deviceModel` |
| 错误码 | `PARAM_INVALID`、`KB_UNAVAILABLE` |
| 访问方式 | 网关以**只读**账号访问 `ai-db`；网关侧不提供任何写入向量库的工具 |

### 7.4 `db.query_device`

| 项 | 内容 |
| --- | --- |
| 入参 | `deviceId: string`（必填） |
| 出参 | `{ model: string; location: string; commissionedAt: string }` |
| 白名单外字段 | `serial_no`（出厂序列号）、`asset_no`（资产编号）、`line_code`（产线编码）、`calib_k`（校准系数）、`owner_dept`、`purchase_price`、`supplier_contact`（裁剪） |
| 说明 | Agent 需要型号是为了构造检索 query；不需要采购价、供应商联系人等商业信息 |

### 7.5 `db.query_workorder`

| 项 | 内容 |
| --- | --- |
| 入参 | `deviceId: string`（必填）、`status?: 'open' \| 'closed' \| 'all'`（默认 `all`）、`limit?: number`（默认 10，上限 30） |
| 出参 | `Array<{ orderNo: string; status: string; handledAt: string \| null }>` |
| 白名单外字段 | `handler_id`、`cost`、`internal_remark`、`customer_complaint`（裁剪） |
| 说明 | 让 Agent 知道"该设备历史上修过什么、修好了没有"，这对诊断有用；但处理人工号、成本、内部备注属于业务敏感信息 |

### 7.6 `report.write`

| 项 | 内容 |
| --- | --- |
| 入参 | `taskId: string`、`deviceId: string`、`summary: string`（≤500）、`evidence: object`（结构化数值依据）、`candidates: Array<{fault: string; confidence: number; source: string}>`、`recommendations: Array<{level: 'immediate'\|'planned'; text: string}>`、`uncertainty?: string` |
| 出参 | `{ reportId: string; createdAt: string }` |
| 写入前的校验 | ① `deviceId ∈ authorizedDevices`；② `taskId` 与令牌一致；③ 字段长度与枚举校验；④ **出参脱敏**：`evidence` 中不得包含白名单外的原始字段名 |
| 唯一性 | 单会话 `report.write` 配额为 1（见 `docs/02` §4.5），防止刷报告 |
| 错误码 | `PARAM_INVALID`、`DEVICE_OUT_OF_SCOPE`、`FORBIDDEN`、`QUOTA_EXCEEDED` |

**注意**：`report.write` 是**唯一**的写工具，且只能写 `agent_report` 一张表。网关不提供工单创建、设备修改、用户管理等写能力给 Agent——这些是人类的职责，走 `business-api`。

### 7.7 工具契约的共享类型

契约定义在 `packages/shared-contracts/`，被网关、插件层、前端三处共用：

```ts
// packages/shared-contracts/src/tools.ts（示意）
export const TOOL_READ_SENSOR = 'mqtt.read_sensor' as const;

export interface ReadSensorParams {
  deviceId: string;
  from: string;
  to: string;
  limit?: number;
}

export interface ReadSensorRow {
  temp: number;
  vibration: number;
  time: string;
}

export const READ_SENSOR_FIELD_WHITELIST = ['temp', 'vibration', 'time'] as const;
```

**单一事实来源**的价值：字段白名单写在一处，网关裁剪、插件层剥离、前端展示三处引用同一常量，避免"改了一处漏了一处"。

---

## 8. 审计落库（fail-closed）

### 8.1 语义

```
任意一次 /tools/invoke（无论放行 / 拒绝 / 收窄）
   → 生成或沿用 traceId
   → 写入 business-db.audit_log
   → 写入失败 ⇒ 返回 AUDIT_UNAVAILABLE，不返回数据
```

**这是本系统最重要的行为约束之一。** 审计不是旁路日志，而是调用链的必经环节。

### 8.2 审计条目

字段定义以 `docs/05-数据库设计.md` 的 `audit_log` 表 DDL 为准（**单一权威**）。网关写入的关键字段：

| 字段（DDL 名） | 网关侧取值 |
| --- | --- |
| `trace_id` | 请求头 `x-trace-id`（缺失则生成） |
| `layer` | `'GATEWAY'` |
| `caller_type` / `caller_id` | 来自 `RequestContext`（`AGENT` / `USER` + 身份） |
| `agent_id` / `session_id` | 来自令牌载荷 |
| `tool_name` | 请求的 `tool` |
| `tool_args_digest` | **脱敏后**的规范化 JSON 的 SHA-256（不存原文） |
| `policy_result` | `ALLOW` / `DENY` / `REWRITE` / `MASK` / `RATE_LIMIT` |
| `policy_rule_id` | 命中的网关规则编号（`G1`-`G9`） |
| `deny_reason` | 拒绝原因文本 |
| `result_fields` | 实际返回给 Agent 的字段名 |
| `redacted_fields` | 被裁剪剥离的字段名 |
| `row_count` / `duration_ms` / `ts` | — |

### 8.3 与插件层审计的关联

插件层（L1）与网关（L2）各写一条审计，**共享同一个 `traceId`**。Web 端的审计查询页可以按 `traceId` 把两条串起来展示，直观呈现"策略层判定 → 边界层执行"的完整链路。这是答辩演示的亮点。

### 8.4 写入性能

- 原型期：同步写入（`better-sqlite3` 同步 API），配合 WAL 模式。
- 若同步写入成为瓶颈：改为**同一请求内的异步写入 + 请求结束前 `await`**，仍然保持 fail-closed 语义（请求不返回即视为失败）。**不允许**改成"先返回数据、后台补写审计"——那会破坏 I4 不变量。
- 落地期（PostgreSQL）：审计表按月分区 + 批量插入。

---

## 9. MQTT 接入

### 9.1 订阅

| 项 | 值 |
| --- | --- |
| Broker | Mosquitto，仅监听内网/本机 |
| 订阅主题 | `iot/+/telemetry` |
| 服务账号 | 网关专用账号，具备 `iot/+/telemetry` 的**读**权限与 `iot/+/cmd` 的**写**权限（落地期） |
| QoS | 1（至少一次，靠 `seq` 去重） |

### 9.2 遥测入库

```
MQTT message
  → 解析 JSON，schema 校验（zod）
  → 校验 deviceId 是否在 device 表中登记（未登记 ⇒ 丢弃 + 记告警）
  → 校验 seq 单调性（重复 ⇒ 幂等丢弃）
  → 校验时间戳合理性（偏离服务器时间超过阈值 ⇒ 标记异常但仍入库）
  → 批量写入 business-db.sensor_data（批量窗口 1s 或 200 条，先到先触发）
```

### 9.3 为什么遥测不落 ai-db

遥测是**业务数据**，属于 `business-db`。`ai-db` 只放向量切片与会话记忆（不变量 I6）。这个边界在实现时容易被"顺手存一份到向量库方便检索"破坏，必须在代码评审中盯住。

### 9.4 下行指令（落地期）

原型阶段**不开放** Agent 直接下发控制指令。落地期若开放，必须：

1. 经网关新增工具 `mqtt.publish_cmd`（而非让 Agent 直连 Broker）；
2. 工具需工程师角色或显式审批（`ASK` 判定）；
3. 指令白名单（改采样率、重启、触发 OTA），不接受任意载荷；
4. 发布前落审计，审计失败即不发布；
5. 终端侧校验指令签名与 `nonce` 防重放（见 `docs/07`）。

---

## 10. 错误码表

| 码 | HTTP | 含义 | 调用方应对 |
| --- | --- | --- | --- |
| `UNAUTHORIZED` | 401 | 令牌缺失/无效/过期 | 重新申请令牌 |
| `FORBIDDEN` | 403 | 身份无权执行该工具 | 不重试，向用户说明 |
| `TOOL_NOT_ALLOWED` | 403 | 工具不在白名单 | 不重试 |
| `PARAM_INVALID` | 400 | 入参 schema 校验失败 | 修正参数后重试一次 |
| `DEVICE_OUT_OF_SCOPE` | 403 | 设备超出授权范围 | **不重试**（这是越权信号） |
| `QUOTA_EXCEEDED` | 429 | 配额超限 | 不重试，收敛输出结论 |
| `RATE_LIMITED` | 429 | 速率超限 | 等待 `retryAfter` 后重试一次 |
| `AUDIT_UNAVAILABLE` | 503 | 审计落库失败 | **不重试**，报告系统异常 |
| `KB_UNAVAILABLE` | 503 | 向量库不可用 | 降级为"无知识库结论"并在报告中说明 |
| `UPSTREAM_TIMEOUT` | 504 | 数据源超时 | 可重试一次 |
| `INTERNAL_ERROR` | 500 | 网关内部异常 | 不重试，记 traceId 上报 |

**设计原则**：错误码必须能让 Agent 区分"改参数重试有用"（`PARAM_INVALID`）与"重试无用且是越权信号"（`DEVICE_OUT_OF_SCOPE`）。这直接决定 ReAct 循环是否收敛。`docs/02` 的提示词塑形第 4 条也向模型明确了这一点。

---

## 11. 限流、健康检查与降级

### 11.1 限流

| 维度 | 原型期阈值 | 超限行为 |
| --- | --- | --- |
| 单 Agent 令牌 | 60 次 / 分钟 | `RATE_LIMITED` |
| 全局 | 200 次 / 分钟 | `RATE_LIMITED` |
| 单工具（`mqtt.read_sensor`） | 20 次 / 分钟 / Agent | `RATE_LIMITED` |

限流在**策略校验之前**执行，避免被限流请求产生大量审计写入。

### 11.2 健康检查

```jsonc
// GET /health
{
  "ok": true,
  "checks": {
    "businessDb": "up",
    "aiDb": "up",
    "mqtt": "up",
    "auditWritable": true      // 关键：审计可写性
  },
  "version": "0.1.0",
  "uptimeSec": 3600
}
```

`auditWritable: false` 时，网关应**主动进入拒绝模式**（所有工具调用返回 `AUDIT_UNAVAILABLE`），而不是继续放行。这保证了"审计完整性优先于可用性"。

### 11.3 降级策略

| 故障 | 降级行为 |
| --- | --- |
| `ai-db` 不可用 | `kb.search` 返回 `KB_UNAVAILABLE`；其他工具正常。Agent 应在报告中说明"知识库不可用，结论仅基于遥测数据" |
| `business-db` 不可用 | 所有读工具返回 `UPSTREAM_TIMEOUT`；**审计也无法写入 ⇒ 整体拒绝服务**（fail-closed） |
| MQTT 不可用 | 遥测停止入库；读工具仍可查历史数据；告警 |
| 审计不可写 | **拒绝所有工具调用**（关键：不允许无审计的数据流出） |

**注意"business-db 不可用"这一行的含义**：由于审计必须落在 `business-db`，业务库故障会导致网关整体拒绝服务。这是 fail-closed 的必然代价。若要求更高的可用性，落地期应把审计表独立到单独实例，使"审计可用"与"业务数据可用"解耦。这一点要写进答辩材料，是"我思考过代价"的证据。

---

## 12. 可观测性

| 项 | 做法 |
| --- | --- |
| `traceId` | 请求入口生成（或沿用 `x-trace-id`），贯穿 L1 审计、L2 审计、响应体、日志 |
| 日志 | 结构化 JSON，字段：`ts`、`level`、`traceId`、`tool`、`actorId`、`decision`、`durationMs` |
| 日志脱敏 | 开启 OpenClaw 侧的 `logging.redactSensitive`；网关侧对参数做摘要而非原文 |
| 指标（可选） | 各工具调用次数/失败率/P95 耗时；`/metrics` 输出 |
| 审计查询 | Web 端审计页按 `traceId` / 时间 / 工具名 / 判定结果筛选 |

---

## 13. 测试要点

| 类型 | 用例 | 期望 |
| --- | --- | --- |
| 单元 | 每个工具的 schema 校验（边界值、缺字段、类型错） | 返回 `PARAM_INVALID` |
| 单元 | 字段裁剪：给一行含全部列的原始数据 | 返回载荷只含白名单键；`meta.redactedFields` 完整 |
| 单元 | 时间窗超限 | 被收窄至上限，审计记录原窗口与实际窗口 |
| 单元 | 条数超限降采样 | 保留首尾 + 极值点 |
| 集成 | 无令牌 / 过期令牌 | `401` + 审计落库 |
| 集成 | 越权设备 | `DEVICE_OUT_OF_SCOPE` + 审计落库 + 不返回任何数据 |
| 集成 | 审计写入失败（只读挂载 business-db） | `AUDIT_UNAVAILABLE`，**不返回数据** |
| 集成 | 速率超限 | `429` 且 `retryAfter` 正确 |
| 集成 | MQTT 上报 → 入库 → 读回 | 端到端一致，`seq` 去重生效 |
| 集成 | 相同 `traceId` 的 L1 与 L2 审计 | 两条记录可按 `traceId` 关联 |
| 安全反证 | 网关进程外（模拟 Agent 环境）尝试连 business-db / ai-db / MQTT | 全部失败，原因为凭证缺失 |
| 性能 | 1000 行遥测写入 / 300 行读取 | 记录 P95 耗时，作为答辩数据 |

---

## 14. 部署要点

- 由 PM2 守护，进程名 `mcp-gateway`，配置见 `deploy/ecosystem.config.cjs`。
- **启动时强校验环境变量**：`BUSINESS_DB_URL`、`AI_DB_URL`、`MQTT_URL`、`AGENT_TOKEN_SECRET` 缺失即**启动失败**（不允许以半配置状态运行）。
- `max_memory_restart` 设置阈值，防止 MQTT 积压导致内存膨胀。
- 日志路径统一到 `logs/mcp-gateway-*.log`，由 PM2 管理。
- 与 OpenClaw 的启动顺序：**先起网关，再起 OpenClaw**（插件层 `plugin:init` 会做网关健康检查，网关不可达则拒绝启用）。

```bash
pm2 start deploy/ecosystem.config.cjs --only mcp-gateway
pm2 start deploy/ecosystem.config.cjs --only openclaw
pm2 save
```
