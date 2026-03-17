Claude code agent 教程
Claude Code 采用**四层记忆层级**，优先级从高到低：
~~~
1. 企业级配置（Enterprise policy）    ← 最高优先级，只读
2. 用户级 CLAUDE.md                   ← ~/.claude/CLAUDE.md，对所有项目生效
3. 项目级 CLAUDE.md                   ← 项目根目录，随 Git 提交共享给团队
4. 子目录级 CLAUDE.md                 ← src/、api/、tests/ 等子目录，按上下文加载
~~~