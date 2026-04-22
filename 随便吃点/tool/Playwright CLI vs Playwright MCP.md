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