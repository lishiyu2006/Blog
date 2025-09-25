<!-- 
  VitePress 允许在 Markdown 文件中直接使用 HTML 和 Vue 组件。
  我们用 HTML 和 CSS 来构建一个自定义的布局。
-->

<!-- 1. 样式定义 (CSS) -->
<style>
/* 页面整体居中，并设置最大宽度 */
.custom-layout {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

/* 顶部个人信息区域 */
.profile-section {
  display: flex;
  align-items: center;
  gap: 2rem; /* 头像和文字的间距 */
  margin-bottom: 4rem;
}

/* 头像样式 (已设为圆角方形) */
.profile-avatar {
  width: 150px;
  height: 150px;
  border-radius: 12px;
  object-fit: cover;
  border: 3px solid var(--vp-c-brand-1);
}

/* 个人信息文本区域 */
.profile-info h1 {
  font-size: 3rem;
  font-weight: 700;
  margin: 0;
  border-bottom: none;
}
.profile-info p {
  color: var(--vp-c-text-2);
  font-size: 1rem;
  line-height: 1.6;
  margin-top: 0.5rem;
}

/* 各个内容区块的标题样式 */
.content-section h2 {
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-top: 3rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 0.5rem;
}
.content-section p, .content-section ul {
  color: var(--vp-c-text-1);
}
.content-section ul {
  padding-left: 1.5rem;
}

/* 技术栈分类样式 */
.skill-category {
  margin-bottom: 1.5rem;
}
.skill-category strong {
  display: inline-block;
  /* 调整标签宽度以适应新标签 */
  width: 130px; 
  font-weight: 600;
}
.tech-icons {
  display: inline-flex;
  gap: 0.5rem;
  vertical-align: middle;
}
.tech-icons img {
  height: 24px;
}
:root.dark .tech-icons img {
  filter: brightness(1.2);
}
</style>

<!-- 2. 页面内容 (HTML) -->
<div class="custom-layout">

  <!-- 顶部个人信息 -->
  <div class="profile-section">
    <!-- 头像 -->
    <!-- 记得把你的 logo 图片放到 public 文件夹下 -->
    <img src="/logo.png" alt="Lizzy的头像" class="profile-avatar">
<!-- 个人介绍 -->

<div class="profile-info">
  <h1>Lizzy</h1>
  <p>大二在读</p>
  <p>主攻方向: 大语言模型 (LLM)</p>
</div>

  </div>

  <!-- 技术栈部分 -->
  <div class="content-section">
    <h2>技术栈</h2>
<div class="skill-category">
  <strong>常用语言:</strong>
  <span class="tech-icons">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </span>
</div>

<div class="skill-category">
  <strong>大语言模型 / AI:</strong>
  <span class="tech-icons">
    <!-- 这里为你预置了几个常用图标，你可以继续添加 -->
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">

  </span>
</div>


  </div>

  <!-- 我的兴趣 -->
  <div class="content-section">
    <h2>我的兴趣</h2>
    <ul>
      <li>探索前沿技术，特别是 <strong>大语言模型（LLM）、Agent 应用和 AIGC</strong>。</li>
      <li>阅读与写作，享受将复杂概念用清晰的语言表达出来的过程。</li>
      <li><strong>音乐、咖啡 和 偶尔的摄影</strong>。</li>
    </ul>
  </div>

  <!-- 联系我 -->
  <div class="content-section">
    <h2>联系我</h2>
    <p>如果你想与我交流，或者对网站内容有任何建议，可以通过以下方式找到我：</p>
    <ul>
      <li><strong>GitHub</strong>: <a href="https://github.com/lishiyu2006" target="_blank">@lizzy</a></li>
      <li><strong>Email</strong>: <a href="mailto:2061105178@qq.com">2061105178@qq.com</a></li>
    </ul>
  </div>

  <br>
  <p style="text-align: center; color: var(--vp-c-text-2);">--- 再次感谢你的来访！ ---</p>

</div>