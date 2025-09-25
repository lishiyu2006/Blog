---
# "---" 这两行之间的内容是 YAML Frontmatter，用于配置页面
# 它不是 Markdown 正文，不会直接显示出来

layout: home # 指定使用 VitePress 的特殊首页布局

# ----------------- 首页主要内容配置 -----------------
hero:
  name: "下午茶" # 你的网站主标题
  text: "一个知识沉淀的角落" # 主标题下的副标题
  tagline: 凡是过去，皆为序章。——莎士比亚 # 标语
  image: # 你可以在这里放一个 logo 或图片
    src: /logo.png # 图片路径 (需要你把 logo.png 放到 public 文件夹下)
    alt: 下午茶 Logo
# --- START: 这是为您新增的样式配置 ---
    # 注意 style 的缩进，它比 image 多2个空格
    style:
      border-radius: 24px
      # 这两行又比 style 多2个空格
      #mask-image: radial-gradient(circle, black 60%, transparent 100%)
  # 注意 actions 的缩进，它必须和 name, text, image 等对齐
# --- END: 这是为您新增的样式配置 ---
  actions:
    - theme: brand # 主按钮 (颜色更突出)
      text: 开始阅读 # 按钮文字
      link: /diary/preface.md # 按钮链接，指向你第一个想让别人看的页面
    - theme: alt # 次要按钮
      text: 关于我
      link: /about # 指向 about.md 页面 (需要你自己创建)

# ----------------- 特色卡片配置 -----------------
features:
  - icon: 📖 # 你可以使用 emoji 作为图标
    title: 学习日记
    details: 我走得很慢，但我从不后退。——林肯
    link: /diary/preface.md# 点击这个卡片跳转的链接
  - icon: 🧠
    title: LLM 探索
    details: 人生的价值，并不是用时间，而是用深度去衡量的。 —— 列夫·托尔斯泰
    link: /LLM/preface.md# 点击这个卡片跳转的链接
  - icon: 🚀
    title: 项目实践
    details: 梦想家命长，实干家寿短。——约·奥赖利
    link: /project/preface.md# 点击这个卡片跳转的链接
---

<!-- 
上面的 Frontmatter 配置会自动生成一个漂亮的首页。
你不需要在这里写任何 Markdown 正文内容。
当然，如果你想在特色卡片下方添加更多内容，可以直接在这里用 Markdown 语法写。
-->