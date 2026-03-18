---
# "---" 这两行之间的内容是 YAML Frontmatter，用于配置页面
# 它不是 Markdown 正文，不会直接显示出来

layout: home # 指定使用 VitePress 的特殊首页布局

# ----------------- 首页主要内容配置 -----------------
hero:
  name: "下午茶" # 你的网站主标题
  text: "一个知识沉淀的角落" # 主标题下的副标题
  tagline: 凡是过去，皆为序章。——莎士比亚 # 标语
  image:
      src: https://github.com/lishiyu2006/Blog/blob/master/logo.png?raw=true
      alt: 下午茶 Logo  # <-- 检查这里！必须比上面的 image 缩进 2 个空格
      border-radius: 999px
      # 这两行又比 style 多2个空格
      #mask-image: radial-gradient(circle, black 60%, transparent 100%)
  # 注意 actions 的缩进，它必须和 name, text, image 等对齐
# --- END: 这是为您新增的样式配置 ---
  actions:
    - theme: brand # 主按钮 (颜色更突出)
      text: 开始阅读 # 按钮文字
      link: /前端/1.preface.md # 按钮链接，指向你第一个想让别人看的页面
    - theme: alt # 次要按钮
      text: 关于我
      link: ./about # 指向 about.md 页面 (需要你自己创建)

# ----------------- 特色卡片配置 -----------------
features:
  - # 你可以使用 emoji 作为图标
    title: 学习日记now
    details: 我走得很慢，但我从不后退。——林肯
    link: /前端/1.preface.md
  - title: 过往
    details: 人生的价值，并不是用时间，而是用深度去衡量的。 —— 列夫·托尔斯泰
    link: /随便吃点/1.preface.md
---

<!-- 
上面的 Frontmatter 配置会自动生成一个漂亮的首页。
你不需要在这里写任何 Markdown 正文内容。
当然，如果你想在特色卡片下方添加更多内容，可以直接在这里用 Markdown 语法写。
-->