# Playwright CLI vs Playwright MCP

## 安装

```bash
npm install -g @playwright/cli@latest
playwright-cli --help
```

## 打开一个可视化仪表盘，查看控制所有正在运行的浏览器

~~~bash
playwright-cli show
~~~
## 指令

~~~bash
playwright-cli open [url]                # 打开浏览器，（可选）跳转到指定的 URL
playwright-cli goto <url>                # 导航/跳转到指定的 URL
playwright-cli close                     # 关闭当前页面
playwright-cli type <text>               # 在可编辑元素中输入文本
playwright-cli click <ref> [button]      # 在网页元素上执行点击操作
playwright-cli dblclick <ref> [button]   # 在网页元素上执行双击操作
playwright-cli fill <ref> <text>         # 向可编辑元素填充文本
playwright-cli fill <ref> <text> --submit # 填充文本并直接按下回车键提交
playwright-cli drag <startRef> <endRef>  # 在两个元素之间执行拖拽操作
playwright-cli hover <ref>               # 悬停（鼠标滑过）在指定元素上
playwright-cli select <ref> <val>        # 在下拉菜单中选择一个选项
playwright-cli upload <file>             # 上传一个或多个文件
playwright-cli check <ref>               # 勾选复选框或单选按钮
playwright-cli uncheck <ref>             # 取消勾选复选框或单选按钮

# 快照与定位相关
playwright-cli snapshot                  # 捕获页面快照以获取元素的引用 ID (ref)
playwright-cli snapshot --filename=f     # 将快照保存到指定的文件中
playwright-cli snapshot <ref>            # 对特定的元素进行快照
playwright-cli snapshot --depth=N        # 限制快照层级深度以提高效率

# 交互与环境相关
playwright-cli eval <func> [ref]         # 在页面或特定元素上执行 JavaScript 表达式
playwright-cli dialog-accept [prompt]    # 接受对话框（如弹窗），可输入提示内容
playwright-cli dialog-dismiss            # 拒绝/关闭对话框
playwright-cli resize <w> <h>            # 调整浏览器窗口的宽度和高度
~~~

## Navigation 导航

```
playwright-cli go-back                  # 返回上一页
playwright-cli go-forward               # 前进到下一页
playwright-cli reload                   # 刷新/重载当前页面
```

## Keyboard 键盘

```
playwright-cli press <key>              # 按一下某个键（如 `a`, `ArrowLeft`, `Enter`）
playwright-cli keydown <key>            # 按住某个键不松开
playwright-cli keyup <key>              # 松开某个键
```

## Mouse 鼠标

```
playwright-cli mousemove <x> <y>        # 移动鼠标到指定的坐标 (x, y)
playwright-cli mousedown [button]       # 按下鼠标按键（可指定左键、右键、中键）
playwright-cli mouseup [button]         # 松开鼠标按键
playwright-cli mousewheel <dx> <dy>     # 滚动鼠标滚轮（dx 为水平，dy 为垂直）
```

## Save as 保存为

```
playwright-cli screenshot [ref]         # 对当前页面或特定元素 [ref] 进行截图
playwright-cli screenshot --filename=f  # 保存截图并指定文件名
playwright-cli pdf                      # 将页面保存为 PDF 格式
playwright-cli pdf --filename=page.pdf  # 保存 PDF 并指定文件名
```

### Tabs 标签页

```
playwright-cli tab-list                 # 列出浏览器中所有的标签页
playwright-cli tab-new [url]            # 新建一个标签页（可选直接打开 URL）
playwright-cli tab-close [index]        # 关闭指定索引的标签页
playwright-cli tab-select <index>       # 切换/选择到指定索引的标签页
```

### Storage & Cookies 存储与 Cookie

```
# Storage State（整体存储状态，常用于保存/恢复登录态）
playwright-cli state-save [filename]    # 保存当前的存储状态（含 Cookie 和本地存储）
playwright-cli state-load <filename>    # 加载已保存的存储状态

# Cookies（缓存数据）
playwright-cli cookie-list [--domain]   # 列出所有 Cookie（可按域名过滤）
playwright-cli cookie-get <name>        # 获取特定名称的 Cookie 值
playwright-cli cookie-set <name> <val>  # 设置一个 Cookie
playwright-cli cookie-delete <name>     # 删除特定名称的 Cookie
playwright-cli cookie-clear             # 清空所有 Cookie

# LocalStorage（本地持久化存储）
playwright-cli localstorage-list        # 列出所有 localStorage 条目
playwright-cli localstorage-get <key>   # 获取指定 key 的 localStorage 值
playwright-cli localstorage-set <k> <v> # 设置 localStorage 的键值对
playwright-cli localstorage-delete <k>  # 删除指定的 localStorage 条目
playwright-cli localstorage-clear       # 清空所有 localStorage

# SessionStorage（会话级存储，关闭标签页后失效）
playwright-cli sessionstorage-list      # 列出所有 sessionStorage 条目
playwright-cli sessionstorage-get <k>   # 获取指定 key 的 sessionStorage 值
playwright-cli sessionstorage-set <k> <v> # 设置 sessionStorage 的键值对
playwright-cli sessionstorage-delete <k>  # 删除指定的 sessionStorage 条目
playwright-cli sessionstorage-clear     # 清空所有 sessionStorage
```