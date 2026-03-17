# Claude code agent 教程
## 一、Claude code的读取顺序

Claude Code 采用**四层记忆层级**，优先级从高到低：

~~~
1. 企业级配置（Enterprise policy）    ← 最高优先级，只读
2. 用户级 CLAUDE.md                   ← ~/.claude/CLAUDE.md，对所有项目生效
3. 项目级 CLAUDE.md                   ← 项目根目录，随 Git 提交共享给团队
4. 子目录级 CLAUDE.md                 ← src/、api/、tests/ 等子目录，按上下文加载
~~~

**具体的规则优先**：子目录的 CLAUDE.md 会覆盖上层的同类规则。

## 二、CLAUDE.md文件示例「在其他的ide，例.cursor里面变成CURSOR.md」

CLAUDE.md的含义是：类似skill，在启动新会话时，Claude code会自己将其注入到系统提示词里，也可以理解成可以被用户配置的长期记忆

Claude 会分析你的目录结构，自动生成一份针对你的技术栈的 CLAUDE.md 骨架。例如，在一个 Node.js 项目中运行 `/init`，Claude 会自动检测框架、测试工具、构建命令等，30 秒内生成一份 80% 完整度的初始文件。「仅限Claude里面有」

~~~cmd
/init
~~~

### 推荐的 CLAUDE.md 结构

~~~md
# 项目约定

## 技术栈
- 前端：Next.js 15、TypeScript 5.7、Tailwind CSS 4
- 后端：Node.js 22、Prisma 6
- 测试：Vitest 3.2

## 代码规范
- 始终使用函数式 React 组件
- 文件名使用 kebab-case
- 测试文件与源码放在同一目录

## 常用命令
- 构建：`pnpm build`
- 测试：`pnpm test`
- 启动开发服务器：`pnpm dev`

## API 约定
- 所有 API 路由以 `/api/v1/` 开头
- 错误响应格式：`{ error: string, code: number }`
~~~
### 写好 CLAUDE.md 的黄金法则

**✅ 要这样写：**

- 使用祈使句和简短列表，而非叙述性段落
- 包含具体的版本号和命令
- 加入代码示例（5 行示例胜过 50 字说明）
- 控制在 **200 行以内**（超过部分不会在会话开始时加载）

**❌ 避免这样写：**

- 模糊指令如"遵循最佳实践"或"写干净的代码"
- 过多通用规则（只放这个项目独有的约定）
- 过时的信息（建议每月审查一次）
