# 06 - Web 前端设计

## 文档定位

本文档定义「基于开源模块化 Agent 的工业预测性运维系统」的 Web 管理界面设计，对应仓库 `apps/web/`。前端是**工程师人机界面**，负责设备状态监视、Agent 分析报告阅读与推理链路复核、审计日志查询、工单闭环、知识库维护与 RBAC 管理。文档给出技术选型与理由、目录结构、页面清单与线框、REST 接口契约、鉴权与权限模型、实时数据方案取舍、构建部署方式，是前端开发的唯一依据，也是答辩中「人机协同 + 可审计」这一主张的界面证据。

需要特别强调本文档贯穿的安全边界：**前端不直连数据库、不直连 MQTT，也不直连 Agent。** 所有数据都经 business-api 或 MCP 网关的统一入口获取，同样受 JWT 鉴权、RBAC 授权与 `audit_log` 审计约束；前端展示的 `traceId` 就是审计表中的 `trace_id`，形成「界面上的每一次操作都可回查审计记录」的闭环。

## 关联文档

- `docs/01-系统总体方案.md` —— 系统分层、部署拓扑
- `docs/02-安全管控插件层设计.md` —— 策略引擎与审计钩子（前端审计页的数据来源）
- `docs/04-MCP安全网关设计.md` —— MCP 工具清单、字段白名单（决定前端能看到哪些字段）
- `docs/05-数据库设计.md` —— 表结构、字段语义、REST 数据形态
- `docs/adr/` —— 前端框架选型、实时方案选型等决策记录

---

## 1. 定位与角色

### 1.1 系统位置

```
┌──────────────────────────────────────────────────────────────┐
│                     工程师浏览器                              │
│         apps/web  (Vue3 + TS + Vite, 纯静态 SPA)             │
└───────────────┬──────────────────────────────────────────────┘
                │ HTTPS + JWT (Authorization: Bearer)
                ▼
┌──────────────────────────────────────────────────────────────┐
│              统一入口层 (Nginx 反向代理)                       │
│   /api/*      → business-api   (业务接口服务)                 │
│   /mcp/*      → mcp-gateway    (MCP 安全网关)                 │
│   /ws/*       → 事件流 (原型期可选)                            │
└───────┬──────────────────────────────────┬───────────────────┘
        │                                  │
        ▼                                  ▼
┌────────────────────┐          ┌──────────────────────────────┐
│  business-api      │          │  mcp-gateway                 │
│  JWT 校验 / RBAC   │          │  身份识别 / 策略 / 字段过滤    │
│  业务查询与写库     │          │  审计落盘                     │
└─────────┬──────────┘          └───────┬──────────────────────┘
          │                             │
          ▼                             ▼
   business-db (业务库)          ai-db (知识库/会话记忆)
                                        ▲
                                        │ MCP 协议
                                 ┌──────┴──────┐
                                 │  Agent 进程  │
                                 │  (OpenClaw)  │
                                 └─────────────┘
```

### 1.2 前端与三方的交互边界

| 交互对象 | 前端是否直连 | 实际通路 | 约束 |
| --- | --- | --- | --- |
| business-db | **否** | business-api 的 REST 接口 | JWT + RBAC 权限点校验，写操作额外记审计 |
| ai-db | **否** | business-api 的知识库管理接口 | 仅知识库管理页使用，不暴露原始向量 |
| MQTT Broker | **否** | 无任何通路；实时数据经 business-api 查询或 SSE 转发 | 浏览器不具备 MQTT 客户端，也无法访问 1883 端口 |
| Agent 进程 | **否** | 触发分析走 business-api → MCP 网关；读报告走 business-api 读 business-db | 前端不能直接向 Agent 发指令，必须经网关过策略 |
| MCP 网关 | 间接 | 部分只读工具查询接口（如审计自查）可经 `/mcp/*` 代理 | 前端调用同样生成 `audit_log`，`caller_type=USER` |

> 设计取舍：为什么不给前端开放 MCP 工具调用？因为 MCP 协议面向 Agent 的 ReAct 循环，返回的是供模型消费的结构化文本，而非为渲染优化的分页数据结构。前端需要的分页、排序、字段裁剪由 business-api 提供更合适。前端唯一可能的例外是「审计日志自查」页面，若复用网关的 `query_audit_log` 工具，可保持与 Agent 完全一致的数据口径，但会多一次策略判定开销——原型期决定审计页走 business-api，**理由是审计页需要联表展示用户名，而网关出于字段白名单不会返回用户名**。

---

## 2. 技术选型表

| 类别 | 选型 | 版本基线 | 选型理由 | 备选方案与取舍 |
| --- | --- | --- | --- | --- |
| 框架 | Vue 3 | 3.4+ | Composition API + `<script setup>` 类型推导友好；生态成熟，毕设资料多 | React 18：类型能力相当，但团队熟悉度与模板可读性弱于 Vue |
| 语言 | TypeScript | 5.4+ | 与后端共用类型定义，接口契约编译期校验，减少联调返工 | 纯 JS：无类型保障，接口字段改名难以发现 |
| 构建 | Vite | 5.x | 冷启动快、HMR 毫秒级；原生 ESM，配置量小 | Webpack：配置复杂，构建慢，无必要 |
| 路由 | Vue Router | 4.x | 官方方案；支持路由元信息承载权限声明，便于守卫统一处理 | — |
| 状态管理 | Pinia | 2.x | 官方推荐，TS 类型推导优于 Vuex；支持按模块拆分 store | Vuex 4：样板代码多，TS 支持弱 |
| UI 组件库 | Element Plus | 2.x | 表格/表单/分页/日期选择器开箱即用，后台管理系统覆盖率最高 | Naive UI：主题定制更灵活但组件文档中文资料略少；原型期优先开发效率 |
| 图表 | ECharts | 5.x | 时序曲线（`line` + `dataZoom`）性能好，10 万点用 `large` 模式可流畅渲染；支持告警标注 | Chart.js：大数据量下性能不足；uPlot：性能更强但 API 与生态偏底层 |
| 架构可视化 | ECharts graph / vis-network（可选） | — | 报告页推理链路、双库隔离示意可用 graph 类型渲染 | AntV G6：能力更强但包体积大，原型期非必需 |
| HTTP 客户端 | Axios | 1.x | 拦截器统一注入 JWT、统一处理 `{code,message,data,traceId}` 包与 401 刷新 | 原生 fetch：需自行封装拦截与刷新逻辑，收益不明显 |
| 实时数据 | 原型期轮询 + SSE 预留 | — | 见第 7 节 | WebSocket：需网关支持事件流，原型期成本高 |
| 代码规范 | ESLint + Prettier + vue-tsc | — | 与 CI 的 `pnpm typecheck` 联动，保证类型零错误 | — |
| 包管理 | pnpm（monorepo workspace） | 9.x | 与后端 `packages/*` 同仓管理，可共享 `packages/shared-types` | npm/yarn：workspace 体验弱于 pnpm |

---

## 3. 目录结构

```
apps/web/
├── index.html
├── vite.config.ts              # 别名 @ → src，代理 /api → business-api
├── tsconfig.json
├── package.json
├── .env.development            # VITE_API_BASE=/api
├── .env.production             # VITE_API_BASE=/api
└── src/
    ├── main.ts                 # createApp + Pinia + Router + ElementPlus + 全局指令
    ├── App.vue
    ├── api/                    # 接口层：一个领域一个文件，纯函数，无业务逻辑
    │   ├── request.ts          # Axios 实例、拦截器、响应包解包、401 刷新
    │   ├── auth.ts             # 登录/登出/刷新/当前用户
    │   ├── device.ts           # 设备列表/详情/状态
    │   ├── sensor.ts           # 时序数据查询（曲线用）
    │   ├── report.ts           # Agent 报告列表/详情/转工单
    │   ├── audit.ts            # 审计日志查询/导出
    │   ├── workorder.ts        # 工单 CRUD 与状态流转
    │   ├── kb.ts               # 知识库文档管理/重建索引/检索测试
    │   └── system.ts           # 用户/角色/权限管理
    ├── views/                  # 页面级组件，与路由一一对应
    │   ├── login/LoginView.vue
    │   ├── dashboard/DashboardView.vue
    │   ├── device/DeviceListView.vue
    │   ├── device/DeviceDetailView.vue
    │   ├── report/ReportListView.vue
    │   ├── report/ReportDetailView.vue
    │   ├── audit/AuditLogView.vue
    │   ├── workorder/WorkOrderListView.vue
    │   ├── workorder/WorkOrderDetailView.vue
    │   ├── kb/KbDocumentView.vue
    │   ├── kb/KbSearchTestView.vue
    │   ├── system/UserManageView.vue
    │   ├── system/RoleManageView.vue
    │   └── error/NotFoundView.vue
    ├── components/             # 可复用组件
    │   ├── layout/AppLayout.vue        # 侧边栏 + 顶栏 + 内容区
    │   ├── layout/AppSidebar.vue
    │   ├── layout/AppHeader.vue
    │   ├── common/PageHeader.vue
    │   ├── common/StatusTag.vue        # 设备/工单/告警状态标签色映射
    │   ├── common/TraceIdLink.vue      # traceId 展示 + 点击跳审计页
    │   ├── chart/LineChart.vue         # ECharts 封装，含 resize 与 dispose
    │   ├── chart/AlertMarkerChart.vue  # 曲线 + 告警阈值标注
    │   ├── report/ReasoningChain.vue   # ReAct 推理链路时间线
    │   ├── report/CitationList.vue     # 知识库引用列表
    │   └── audit/AuditDetailDrawer.vue # 审计记录详情抽屉
    ├── stores/                 # Pinia
    │   ├── user.ts             # 用户信息、token、权限码集合
    │   ├── device.ts           # 设备列表缓存、当前设备
    │   └── app.ts              # 主题、侧边栏折叠、全局 loading
    ├── router/
    │   ├── index.ts            # 路由表
    │   └── guard.ts            # 登录守卫 + 权限守卫
    ├── types/                  # 与后端共享的接口类型
    │   ├── api.d.ts            # ApiResponse<T>、分页结构
    │   ├── device.d.ts
    │   ├── sensor.d.ts
    │   ├── report.d.ts
    │   ├── audit.d.ts
    │   ├── workorder.d.ts
    │   └── kb.d.ts
    ├── utils/
    │   ├── storage.ts          # token 存储封装（内存 + sessionStorage）
    │   ├── format.ts           # 时间/数值/单位格式化
    │   ├── permission.ts       # hasPermission(code) 判定
    │   └── constants.ts        # 状态枚举与中文映射
    └── directives/
        └── permission.ts       # v-permission 按钮级权限指令
```

---

## 4. 页面清单与线框描述

### 4.1 页面总览

| 页面 | 路由 | 目标用户 | 核心权限码 |
| --- | --- | --- | --- |
| 登录页 | `/login` | 全部 | — |
| 设备总览看板 | `/dashboard` | 工程师 / 管理员 | `device:read` |
| 设备详情 | `/device/:id` | 工程师 | `device:read` `sensor_data:read` |
| Agent 报告列表 | `/report` | 工程师 | `agent_report:read` |
| Agent 报告详情 | `/report/:id` | 工程师 | `agent_report:read` |
| 审计日志查询 | `/audit` | 管理员 / 安全审计员 | `audit_log:read` |
| 工单列表 | `/workorder` | 工程师 | `work_order:read` |
| 工单详情 | `/workorder/:id` | 工程师 | `work_order:read` `work_order:write` |
| 知识库管理 | `/kb` | 工程师 / 管理员 | `kb:read` `kb:write` |
| 知识库检索测试 | `/kb/search` | 工程师 | `kb:read` |
| 用户管理 | `/system/user` | 管理员 | `user:manage` |
| 角色管理 | `/system/role` | 管理员 | `role:manage` |

### 4.2 登录页 `/login`

- **目标用户**：全部。
- **核心交互**：用户名 + 密码 → `POST /api/auth/login` → 返回 `accessToken` / `refreshToken` / 用户信息 / 权限码集合 → 写入 Pinia 与 sessionStorage → 跳转 `/dashboard`；连续失败 5 次后由后端返回锁定提示（对应 `user_account.failed_count`）。
- **调用接口**：`POST /api/auth/login`。
- **安全说明**：前端不做任何密码强度校验以外的加密处理，密码明文走 HTTPS 提交，哈希在服务端用 argon2id 完成；**前端不存密码、不存哈希**。

```
┌──────────────────────────────────────────────┐
│                                              │
│       工业预测性运维系统                       │
│       Predictive Maintenance Platform        │
│                                              │
│      ┌────────────────────────────┐          │
│      │ 用户名  [________________] │          │
│      └────────────────────────────┘          │
│      ┌────────────────────────────┐          │
│      │ 密码    [________________] │          │
│      └────────────────────────────┘          │
│                                              │
│           [      登  录      ]                │
│                                              │
│       ⓘ 本系统所有操作均被审计记录             │
└──────────────────────────────────────────────┘
```

### 4.3 设备总览看板 `/dashboard`

- **目标用户**：工程师、管理员。
- **核心交互**：卡片统计（在线/离线/故障设备数、今日告警数、待处理工单数）；设备状态表格，点击行进入详情；最近 Agent 报告滚动列表；可选「手动触发巡检」按钮 → `POST /api/agent/analyze`（原型期同步返回，落地期改异步任务 + 轮询结果）。
- **调用接口**：`GET /api/devices?page&size`、`GET /api/devices/stats`、`GET /api/agent/reports?size=5`、`GET /api/work-orders/stats`。

```
┌────────────────────────────────────────────────────────────────┐
│ ☰  工业预测性运维系统          🔔3   admin ▾                    │
├──────────┬─────────────────────────────────────────────────────┤
│ 总览看板 │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│ 设备管理 │  │在线 5  │ │离线 1  │ │告警 3  │ │工单 2  │        │
│ 分析报告 │  └────────┘ └────────┘ └────────┘ └────────┘        │
│ 审计日志 │  ┌───────────────────────────────────────────────┐  │
│ 工单管理 │  │ 设备编码    名称    型号   状态   最近上报     │  │
│ 知识库   │  │ STM32-01   主轴电机  NEMA23 ●在线  12s前      │  │
│ 系统设置 │  │ STM32-02   传送带轮  ...    ●在线  15s前      │  │
│          │  │ STM32-03   液压泵    ...    ▲告警  8s前       │  │
│          │  └───────────────────────────────────────────────┘  │
│          │  ┌───────────────────────────────────────────────┐  │
│          │  │ 最新 Agent 报告                    [查看全部] │  │
│          │  │ · STM32-03 振动异常升高，疑似轴承磨损 (HIGH)  │  │
│          │  │ · STM32-01 温度平稳，无需干预 (LOW)          │  │
│          │  └───────────────────────────────────────────────┘  │
└──────────┴─────────────────────────────────────────────────────┘
```

### 4.4 设备详情 `/device/:id`

- **目标用户**：工程师。
- **核心交互**：顶部设备档案卡；中部双曲线图（温度、振动），支持时间范围切换（近 1h / 6h / 24h / 7d）、`dataZoom` 缩放、阈值线标注、告警点标记；曲线下方为该设备告警与工单时间线；「请求 Agent 分析」按钮触发分析并跳转报告页。
- **调用接口**：`GET /api/devices/:id`、`GET /api/sensor-data?deviceId&metric&from&to&interval`、`GET /api/devices/:id/alerts`、`POST /api/agent/analyze`。
- **性能取舍**：原型期直接查原始点（1h 内约 720 点，可接受）；7d 视图改走降采样接口 `interval=5m`（对应 `docs/05-数据库设计.md` 的 `sensor_data_1m` 规划）。

```
┌────────────────────────────────────────────────────────────────┐
│ ← 返回   DEV-003 液压泵          ●在线  型号:HYD-200      │
├────────────────────────────────────────────────────────────────┤
│ [近1h] [近6h] [近24h] [近7d]              [请求 Agent 分析]     │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ 温度 ℃                                              ⚠ 告警 │ │
│ │ 90 ────────────────────────────────────────── 阈值 85      │ │
│ │    ╱╲    ╱╲                                                │ │
│ │ 60 ───────────────────────────────────────────             │ │
│ │         ▲告警                                               │ │
│ └────────────────────────────────────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ 振动 mm/s                                                   │ │
│ │ 8 ───────────────────────────────────────── 阈值 7         │ │
│ │      ╱╲╱╲╱╲                                                 │ │
│ │ 4 ─────────────────────────────────────────                │ │
│ └────────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────┤
│ 告警时间线:  09:12 温度超阈值(CRITICAL)  → 关联工单 WO-...0001 │
│             08:40 振动偏高(WARN)                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.5 Agent 分析报告列表 / 详情

**列表 `/report`**：按设备、严重度、时间范围筛选；表格列含报告号、设备、结论摘要、置信度、严重度、是否转工单、`traceId`。
**调用接口**：`GET /api/agent/reports?deviceId&severity&from&to&page&size`。

**详情 `/report/:id`** 是本系统最核心的展示页，需同时呈现「结论」与「依据」，让工程师能复核 Agent 的判断而非盲信：

- 报告头：结论、置信度、严重度、生成模型、`traceId`（点击跳审计页）。
- **推理链路（ReAct）**：时间线展示每一步 `Thought → Action(tool_name) → Observation`，其中 Action 可展开显示入参摘要、Observation 显示被字段白名单过滤后的返回摘要；被策略拒绝的步骤以红色标出并显示 `deny_reason`。
- 知识库引用：`citations` 列表，点击跳知识库检索测试页并高亮对应切片。
- 操作区：`转工单`（`POST /api/work-orders`，携带 `reportId`）、`归档`。

```
┌────────────────────────────────────────────────────────────────┐
│ 报告 RPT-20260915-0003        [转工单] [归档]                  │
│ 设备: DEV-003 液压泵   严重度: HIGH   置信度: 0.82        │
│ 结论: 振动频谱中 2 倍频分量显著上升，结合温度缓慢爬升，          │
│       判断为轴承早期磨损，建议 72 小时内安排停机检查。          │
│ traceId: 7f3a9c21-...  [查看审计记录 →]                        │
├────────────────────────────────────────────────────────────────┤
│ 推理链路 (ReAct)                                                │
│ ① Thought  需要先确认该设备最近 24h 的振动与温度趋势            │
│    Action   mqtt.read_sensor {deviceId, from, to, limit:200}   │
│                                                 ✅ ALLOW       │
│    Observe  {points:200, max:7.9, trend:+38%}                  │
│ ② Thought  趋势确认异常，需检索该型号的轴承故障手册             │
│    Action   kb.search {query:"液压泵 轴承 振动 2倍频",          │
│             deviceModel:"HYD-200", topK:5}       ✅ ALLOW       │
│    Observe  命中 3 片: 《HYD-200 维护手册》P.42 ...             │
│ ③ Thought  尝试读取该设备的历史维修成本以评估优先级             │
│    Action   db.query_workorder_cost {...}        ❌ DENY        │
│             reason: TOOL_NOT_ALLOWED（工具不在可见清单内）      │
│    策略拦截: 工具不在白名单内 (rule: TOOL_NOT_ALLOWED)          │
├────────────────────────────────────────────────────────────────┤
│ 知识库引用: [1] HYD-200维护手册 P.42 §3.2  [2] 轴承磨损案例-07  │
└────────────────────────────────────────────────────────────────┘
```

> 答辩亮点：第 ③ 步被策略拦截并完整记录在 `audit_log`（`policy_result=DENY`），前端直接展示拦截原因。这把「安全管控插件层真的在起作用」变成了肉眼可见的证据，而不是文档里的承诺。

### 4.6 审计日志查询页 `/audit`

- **目标用户**：管理员、安全审计员。
- **核心交互**：多条件筛选（`trace_id`、时间范围、`tool_name`、`caller_type`、`policy_result`、`session_id`）；表格列含时间、traceId、调用方、agent_id、工具名、策略结果、耗时；行点击打开右侧抽屉展示完整详情（入参摘要、出参字段白名单、命中规则、拒绝原因）；支持导出 CSV（`audit_log:export` 权限，导出动作本身也记审计）。
- **调用接口**：`GET /api/audit-logs?traceId&toolName&policyResult&from&to&page&size`、`GET /api/audit-logs/:id`、`GET /api/audit-logs/export`。
- **说明**：审计页只读，前端不提供任何编辑/删除入口；后端也不暴露 DELETE 接口。

```
┌────────────────────────────────────────────────────────────────┐
│ 审计日志                        [导出 CSV]                      │
├────────────────────────────────────────────────────────────────┤
│ traceId [__________] 工具 [▾全部] 结果 [▾全部]                 │
│ 时间 [2026-09-15 00:00] ~ [2026-09-15 23:59]      [查询][重置] │
├────────────────────────────────────────────────────────────────┤
│ 时间      traceId   层    调用方  工具              结果  耗时 │
│ 09:14:02  7f3a9c21  PLUGIN AGENT  mqtt.read_sensor  ALLOW 2ms  │
│ 09:14:02  7f3a9c21  GATEWAY AGENT mqtt.read_sensor  ALLOW 42ms │
│ 09:14:05  7f3a9c21  PLUGIN AGENT  kb.search         ALLOW 1ms  │
│ 09:14:05  7f3a9c21  GATEWAY AGENT kb.search         ALLOW 118ms│
│ 09:14:06  7f3a9c21  PLUGIN AGENT  db.query_wo_cost  DENY  1ms ⚠│
│ 09:15:11  a1b2c3d4  GATEWAY USER  work_order:create ALLOW 23ms │
├────────────────────────────────────────────────────────────────┤
│                       < 1 2 3 ... 20 >   共 1,842 条           │
└────────────────────────────────────────────────────────────────┘
```

### 4.7 工单管理 `/workorder`

- **目标用户**：工程师。
- **核心交互**：列表按状态/优先级筛选；详情页支持状态流转（`CREATED → ASSIGNED → PROCESSING → RESOLVED → VERIFIED → CLOSED`），非法迁移由后端返回 409 并前端提示；可关联告警与 Agent 报告，展示 `traceId` 形成「告警→分析→工单→闭环」的完整链路。
- **调用接口**：`GET /api/work-orders`、`GET /api/work-orders/:id`、`PATCH /api/work-orders/:id/status`、`POST /api/work-orders`。

```
┌────────────────────────────────────────────────────────────────┐
│ 工单管理          状态[▾全部] 优先级[▾全部]     [+ 新建工单]   │
├────────────────────────────────────────────────────────────────┤
│ 工单号          设备       标题              优先级 状态  处理人│
│ WO-...0001      STM32-03   轴承磨损检查      URGENT 处理中 张三 │
│ WO-...0002      STM32-01   温度传感器校准    MEDIUM 待指派  —   │
├────────────────────────────────────────────────────────────────┤
│ 详情: 关联告警 AL-0231 | 关联报告 RPT-...0003 | traceId ↗     │
│ 状态:  CREATED ─●─ ASSIGNED ─●─ PROCESSING ─○─ RESOLVED ...  │
│        [开始处理] [标记已解决] [取消工单]                      │
└────────────────────────────────────────────────────────────────┘
```

### 4.8 知识库管理 `/kb` 与检索测试 `/kb/search`

- **目标用户**：工程师、管理员。
- **核心交互**：
  - 文档列表：展示 `doc_code`、标题、类型、适用型号、版本、切片数、索引状态；操作含「重建索引」（`POST /api/kb/documents/:id/reindex`）、「删除」（同时清理向量库切片）。
  - 上传文档（原型期可选）：`POST /api/kb/documents`（multipart），入库后状态为 `PENDING`，需手动触发重建。
  - 检索测试页：输入 query、选设备型号与 `top_k`，展示命中切片、相似度分数、来源（文档 / 章节 / 页码），用于验证 RAG 效果并调试切片策略。
- **调用接口**：`GET /api/kb/documents`、`POST /api/kb/documents`、`POST /api/kb/documents/:id/reindex`、`DELETE /api/kb/documents/:id`、`POST /api/kb/search`。

```
┌────────────────────────────────────────────────────────────────┐
│ 知识库管理                              [+ 上传文档]           │
├────────────────────────────────────────────────────────────────┤
│ 文档编号  标题            类型   型号     版本 切片 状态  操作 │
│ KB-001   HYD-200维护手册 manual HYD-200  v2  148  READY [重建]│
│ KB-002   轴承磨损案例集  case   -        v1   92  READY [重建]│
├────────────────────────────────────────────────────────────────┤
│ 检索测试: [液压泵 振动 2倍频        ] 型号[HYD-200▾] k[5▾][检索]│
│ ① 0.71 《HYD-200维护手册》P.42 §3.2 振动异常诊断              │
│ ② 0.64 《轴承磨损案例集》案例07 2倍频特征与轴承间隙           │
└────────────────────────────────────────────────────────────────┘
```

### 4.9 系统设置：用户管理 / 角色管理

- **目标用户**：管理员。
- **核心交互**：用户列表增删改、启用/禁用、重置密码（后端生成一次性初始密码，前端只显示一次）；角色管理为角色分配权限点（复选树），对应 `role_permission` 表；禁止删除内置角色。
- **调用接口**：`GET/POST/PATCH /api/system/users`、`POST /api/system/users/:id/reset-password`、`GET/POST/PATCH /api/system/roles`、`GET /api/system/permissions`。
- **安全说明**：用户管理页的所有写操作都记录审计（`caller_type=USER`），且**不提供任何查看 Agent 数据库配置的入口**——前端根本无法展示它不存在的东西。

---

## 5. 接口约定

### 5.1 统一响应包

```ts
// src/types/api.d.ts
export interface ApiResponse<T> {
  code: number;      // 0 成功；非 0 为业务错误码
  message: string;   // 人类可读提示，可直接 toast
  data: T;           // 业务数据
  traceId: string;   // 全链路追踪 id，与 audit_log.trace_id 同源
}

export interface PageResult<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
}

// 错误码约定
// 0      成功
// 40100  未认证 / token 过期
// 40300  无权限（RBAC 拒绝）
// 40400  资源不存在
// 40900  状态冲突（如工单非法流转）
// 42900  请求过频（网关限流）
// 50000  服务端异常
```

Axios 拦截器职责：请求侧注入 `Authorization: Bearer <accessToken>`；响应侧解包 `data`、`code !== 0` 统一 toast 并抛出、`40100` 触发静默刷新、把 `traceId` 写入响应头透出的调试字段供页面展示。

### 5.2 REST 接口清单

| 方法 | 路径 | 鉴权 | 权限码 | 说明 |
| --- | --- | --- | --- | --- |
| POST | `/api/auth/login` | 否 | — | 登录，返回双 token + 权限码集合 |
| POST | `/api/auth/refresh` | refreshToken | — | 刷新 accessToken |
| POST | `/api/auth/logout` | 是 | — | 登出，服务端失效 refreshToken |
| GET | `/api/auth/me` | 是 | — | 当前用户信息与权限码 |
| GET | `/api/devices` | 是 | `device:read` | 设备分页列表 |
| GET | `/api/devices/stats` | 是 | `device:read` | 看板统计 |
| GET | `/api/devices/:id` | 是 | `device:read` | 设备详情 |
| GET | `/api/devices/:id/alerts` | 是 | `device:read` | 设备告警时间线 |
| GET | `/api/sensor-data` | 是 | `sensor_data:read` | 时序查询，参数 `deviceId/metric/from/to/interval/limit` |
| GET | `/api/agent/reports` | 是 | `agent_report:read` | Agent 报告列表 |
| GET | `/api/agent/reports/:id` | 是 | `agent_report:read` | 报告详情（含推理链路与引用） |
| POST | `/api/agent/analyze` | 是 | `agent_report:trigger` | 触发一次 Agent 分析（经网关） |
| GET | `/api/audit-logs` | 是 | `audit_log:read` | 审计查询，参数 `traceId/toolName/policyResult/callerType/from/to` |
| GET | `/api/audit-logs/:id` | 是 | `audit_log:read` | 审计详情 |
| GET | `/api/audit-logs/export` | 是 | `audit_log:export` | 导出 CSV，本身也记审计 |
| GET | `/api/work-orders` | 是 | `work_order:read` | 工单列表 |
| GET | `/api/work-orders/:id` | 是 | `work_order:read` | 工单详情 |
| POST | `/api/work-orders` | 是 | `work_order:write` | 新建工单（可带 `reportId`） |
| PATCH | `/api/work-orders/:id/status` | 是 | `work_order:write` | 状态流转，非法迁移 409 |
| GET | `/api/kb/documents` | 是 | `kb:read` | 知识库文档列表 |
| POST | `/api/kb/documents` | 是 | `kb:write` | 上传文档 |
| POST | `/api/kb/documents/:id/reindex` | 是 | `kb:write` | 重建索引（异步任务） |
| DELETE | `/api/kb/documents/:id` | 是 | `kb:write` | 删除文档及其切片 |
| POST | `/api/kb/search` | 是 | `kb:read` | 检索测试，参数 `query/deviceModel/topK` |
| GET | `/api/system/users` | 是 | `user:manage` | 用户列表 |
| POST | `/api/system/users` | 是 | `user:manage` | 新建用户 |
| PATCH | `/api/system/users/:id` | 是 | `user:manage` | 修改用户/启停 |
| POST | `/api/system/users/:id/reset-password` | 是 | `user:manage` | 重置密码 |
| GET | `/api/system/roles` | 是 | `role:manage` | 角色列表（含权限点） |
| POST | `/api/system/roles` | 是 | `role:manage` | 新建角色 |
| PATCH | `/api/system/roles/:id/permissions` | 是 | `role:manage` | 调整角色权限 |

**契约说明**：接口路径与字段命名以 `docs/05-数据库设计.md` 的表结构为基准，禁止前端直接拼接 SQL 语义的查询参数（如 `orderBy=xxx`），排序与过滤字段由后端白名单控制，防止通过排序参数探测表结构。

---

## 6. 鉴权与安全

### 6.1 JWT 存放与刷新策略

| 项目 | 方案 | 理由 |
| --- | --- | --- |
| accessToken | 存内存（Pinia）+ `sessionStorage` 兜底 | 避免 `localStorage` 长期驻留被 XSS 一次性窃取；关闭标签页即失效 |
| refreshToken | 由后端下发为 **HttpOnly + Secure + SameSite=Strict** Cookie | JS 不可读，从根本上防 XSS 窃取 |
| accessToken 有效期 | 15 分钟 | 短有效期限制泄露后的可利用窗口 |
| refreshToken 有效期 | 7 天，服务端可吊销 | 兼顾体验与安全 |
| 刷新时机 | 响应 `code=40100` 时拦截器自动调用 `/api/auth/refresh`，成功后重放原请求；**并发请求共享同一个刷新 Promise**，避免刷新风暴 | 防止刷新竞态导致误登出 |
| 刷新失败 | 清空本地状态并跳登录页，携带 `redirect` 参数 | 用户体验与安全兼顾 |

### 6.2 路由守卫

```ts
// src/router/guard.ts 逻辑要点
router.beforeEach(async (to) => {
  // 1. 白名单放行
  if (to.meta.public) return true;

  // 2. 无 token → 登录页，记录 redirect
  const userStore = useUserStore();
  if (!userStore.accessToken) {
    return { name: 'login', query: { redirect: to.fullPath } };
  }

  // 3. 首次进入拉取用户信息与权限码
  if (!userStore.loaded) await userStore.fetchMe();

  // 4. 路由级权限：meta.permission 与用户权限码集合取交集
  if (to.meta.permission && !hasPermission(to.meta.permission)) {
    return { name: 'forbidden' };   // 403 页面，不泄露目标页面内容
  }
  return true;
});
```

路由元信息示例：

```ts
{
  path: 'audit',
  name: 'audit',
  component: () => import('@/views/audit/AuditLogView.vue'),
  meta: { title: '审计日志', permission: 'audit_log:read', icon: 'Document' },
}
```

### 6.3 按钮级权限 `v-permission`

```ts
// src/directives/permission.ts
import type { Directive } from 'vue';
import { useUserStore } from '@/stores/user';

export const vPermission: Directive<HTMLElement, string | string[]> = {
  mounted(el, binding) {
    const store = useUserStore();
    const need = Array.isArray(binding.value) ? binding.value : [binding.value];
    const ok = need.some((code) => store.permissions.includes(code));
    if (!ok) el.parentNode?.removeChild(el);   // 直接移除，而非仅隐藏
  },
};
```

使用方式：

```vue
<el-button v-permission="'work_order:write'" type="primary">新建工单</el-button>
<el-button v-permission="'audit_log:export'">导出 CSV</el-button>
```

**关键原则：前端权限控制只用于提升体验，绝不能作为安全边界。** 后端每个接口都必须独立做 RBAC 校验；即便有人绕过前端直接构造请求，也会在 business-api 层被拒绝并记入审计。

### 6.4 traceId 透传与展示

| 环节 | 行为 |
| --- | --- |
| 后端生成 | 每个请求在 business-api 入口生成或透传 `traceId`，写入响应包 `traceId` 字段 |
| 前端保存 | Axios 响应拦截器将 `traceId` 存入响应对象，页面可按需展示 |
| 展示组件 | `<TraceIdLink :trace-id="x" />`：短哈希 + hover 显示完整值，点击跳 `/audit?traceId=xxx` |
| 价值 | 用户在报告页/工单页看到异常，可一键跳到审计页查看该链路的所有工具调用与策略判定，形成端到端可追溯 |

### 6.5 前端安全清单

- 前端**不持有任何数据库连接信息**、**不持有 MQTT 地址**、**不持有 Agent 内部配置**；构建产物中不出现任何密钥。
- 所有配置来自 `VITE_*` 环境变量，且仅允许公开信息（API 基址、版本号），密钥类配置一律留在服务端。
- 富文本渲染（Agent 报告结论可能含 Markdown）统一用 `DOMPurify` 清洗后再插入，防 XSS。
- 禁用 `v-html` 直插未清洗内容；ESLint 规则 `vue/no-v-html` 设为 error。
- 所有写操作按钮在提交期间 `loading` 锁，防重复提交（后端另有幂等键兜底）。

---

## 7. 实时数据展示方案

### 7.1 方案对比与取舍

| 方案 | 实现成本 | 实时性 | 服务端压力 | 适用场景 | 原型期结论 |
| --- | --- | --- | --- | --- | --- |
| 轮询（setInterval + REST） | 极低 | 1~5s 延迟 | 中（每客户端 N 次/分） | 看板统计、设备状态 | **采用**，看板 10s、设备详情 3s |
| SSE（Server-Sent Events） | 低 | 秒级 | 低（单连接长驻） | 单向推送：新告警、新报告 | **采用**，用于「新告警/新报告」通知条 |
| WebSocket | 中 | 毫秒级 | 低（双向） | 双向交互、Agent 推理流式输出 | **暂不采用**，留待落地期 |
| MQTT over WebSocket | 高 | 毫秒级 | 低 | 浏览器直连 MQTT | **明确不采用**：违反「前端不直连 MQTT」的安全约束 |

### 7.2 原型期方案

1. **轮询**：
   - 看板统计 `/api/devices/stats`、`/api/work-orders/stats` 每 10s 轮询；
   - 设备详情曲线每 3s 追加最新点（只请求 `from=最后一点时间` 的增量区间，避免全量重查）；
   - 页面 `onUnmounted` 必须清理定时器，路由切换不残留。
2. **SSE**：`GET /api/events/stream`（`EventSource`），事件类型 `alert.created`、`report.created`；前端在顶栏铃铛处累加未读数并弹出通知。SSE 天然支持断线重连（浏览器内置 `Last-Event-ID` 重放）。

> 取舍说明：为什么原型期不直接上 WebSocket？因为 WebSocket 需要额外的连接鉴权、心跳、重连、多实例广播（Redis Pub/Sub）等配套，而这些与毕设核心（安全管控插件层）无关。轮询 + SSE 已能完整演示「实时感知」能力，且服务端实现量小、可审计（轮询请求同样记 `traceId`）。

### 7.3 后续接入 OpenClaw Gateway 事件流的路径

OpenClaw 若提供 Gateway 事件流（形如 `agent`、`health` 等事件），可平滑升级：

| 阶段 | 前端行为 | 后端配套 |
| --- | --- | --- |
| 阶段一（原型期） | 轮询 + SSE | business-api 提供 `/api/events/stream`，内部从数据库变更或内存事件总线推送 |
| 阶段二 | 把 SSE 源替换为 WebSocket `/ws/events`，事件类型扩展为 `agent.thought`、`agent.action`、`agent.observation`、`agent.done`、`health` | 网关增加 WebSocket 端点，把 Agent 的 ReAct 步骤作为事件广播 |
| 阶段三 | 报告详情页由「读库渲染完整链路」改为「先流式渲染实时链路，结束后落库固化」 | 网关与 Agent 之间增加事件转发，落库仍走 business-api |

前端封装建议：定义统一的 `useEventStream()` composable，内部屏蔽 SSE 与 WebSocket 差异，页面只订阅 `on('agent.observation', handler)`，将来切换传输层无需改页面代码。事件 payload 中同样携带 `traceId`，保持与审计链一致。

---

## 8. 构建与部署

### 8.1 环境变量

```bash
# apps/web/.env.development
VITE_API_BASE=/api
VITE_APP_TITLE=工业预测性运维系统
VITE_POLL_INTERVAL_DASHBOARD=10000
VITE_POLL_INTERVAL_SENSOR=3000

# apps/web/.env.production
VITE_API_BASE=/api
VITE_APP_TITLE=工业预测性运维系统
```

说明：`VITE_API_BASE` 是前端唯一需要感知的后端地址配置，生产环境建议用相对路径 `/api` 并交由 Nginx 反向代理，避免跨域与硬编码域名。**严禁在前端环境变量中放置任何密钥、数据库连接串或 MQTT 地址。**

### 8.2 构建

```bash
pnpm install
pnpm --filter web typecheck      # vue-tsc --noEmit，CI 必过
pnpm --filter web build          # 产出 apps/web/dist/
pnpm --filter web preview        # 本地预览产物
```

构建产物为纯静态文件（`index.html` + `assets/*`），不含任何服务端逻辑。

### 8.3 部署方式对比

| 方式 | 适用场景 | 配置要点 | 结论 |
| --- | --- | --- | --- |
| Nginx 静态托管 + 反向代理 | 生产推荐 | `root /var/www/web; try_files $uri $uri/ /index.html;`（SPA history 模式必须）；`location /api/ { proxy_pass http://127.0.0.1:7800; }`；开启 gzip / brotli | **推荐**，性能与可控性最好 |
| PM2 托管静态文件 | 原型期省事 | 用 `pm2 serve dist 5173 --spa`；但缺少反向代理与压缩能力，需额外配置代理 | 原型期可用，落地期替换为 Nginx |
| Vite dev server | 仅开发 | 内置 proxy 到 business-api，支持 HMR | 不用于任何对外演示之外的场景 |

### 8.4 部署拓扑示例

```
                      ┌──────────────────────┐
   浏览器 ──HTTPS──▶  │  Nginx :443          │
                      │  ├─ /        → dist/ │
                      │  ├─ /api/    → :7800 │  business-api
                      │  ├─ /mcp/    → :7801 │  mcp-gateway
                      │  └─ /ws/     → :7802 │  事件流(可选)
                      └──────────────────────┘
                                  │
                      ┌───────────┴───────────┐
                      │ 内网：DB / MQTT / Agent│
                      │ 前端不可达，无凭据     │
                      └───────────────────────┘
```

### 8.5 CI 检查项

| 检查 | 命令 | 目的 |
| --- | --- | --- |
| 类型检查 | `pnpm --filter web typecheck` | 保证接口契约变更被编译期发现 |
| Lint | `pnpm --filter web lint` | 统一风格；`vue/no-v-html` 防 XSS |
| 构建 | `pnpm --filter web build` | 确保可发布 |
| 密钥扫描 | 对 `dist/` 与 `.env*` 做正则扫描（`postgres://`、`mqtt://`、`business.db`） | **验证前端产物中不含任何数据库/MQTT 凭据**，与 `docs/05-数据库设计.md` 的 `db:verify` 自检呼应 |

最后一项是本文档与数据库设计文档的呼应点：前端产物的密钥扫描 + 后端的双库隔离自检，共同构成「Agent 无 business-db 权限、前端无任何直连凭据」的双向可执行证明。
