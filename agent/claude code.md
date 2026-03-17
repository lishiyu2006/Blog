# Claude code 教程

## Claude code 就是 agent

## 一、Claude code读取的顺序

Claude Code 采用**四层记忆层级**，优先级从高到低：

~~~md
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

### 子目录 CLAUDE.md

~~~md
my-project/
├── CLAUDE.md              # 全局项目规范
├── src/
│   └── CLAUDE.md          # 仅在处理 src/ 文件时加载
├── api/
│   └── CLAUDE.md          # API 特定约定
└── tests/
    └── CLAUDE.md          # 测试特定规则
~~~~
Claude Code 只在处理对应目录的文件时加载子目录的 CLAUDE.md，节省 token 的同时提供更精准的上下文。

## 三、Auto Memory的原理

Auto Memory 让 Claude 能够跨会话**自我积累知识**，无需用户手动编写任何内容，以便在new session的时候可以带着上次的记忆。Claude 会在工作过程中自动保存笔记，包括：

- 构建命令和调试技巧
- 架构决策笔记
- 代码风格偏好
- 工作流习惯

Claude 并不会每次都保存内容，它会判断哪些信息在**未来会话中有用**才写入。

### 自动记忆的文件结构

~~~md
~/.claude/projects/<project>/memory/
├── MEMORY.md          # 简洁的索引文件，每次会话开始时加载（前 200 行）
├── debugging.md       # 调试模式的详细笔记
├── api-conventions.md # API 设计决策
└── ...                # Claude 创建的其他主题文件
~~~

<>代表占位符，这里是项目真正的名字，除此之外的都是固定的，这样才能被软件固定读取

### 触发自动记忆

当你告诉 Claude 某些事情时，它会自动保存到记忆中：

```cmd
你：始终使用 pnpm，不要用 npm
你：记住 API 测试需要本地运行 Redis 实例
你：我们的日期格式统一用 ISO 8601
```

**想保存到 CLAUDE.md 而不是 Auto Memory？** 明确说明：

```cmd
你：把这条加到 CLAUDE.md
```

### `#` 快捷键——快速添加记忆

这是一个隐藏的效率神器：

# 始终在函数参数中使用具名参数（named parameters）

按下 `#` 键，输入你想记住的内容，按回车——Claude Code 会自动将其写入对应的 CLAUDE.md 文件。非常适合：

- 记录项目约定
- 保存常用 Bash 命令
- 记下代码风格细节

