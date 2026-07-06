import { defineConfig } from 'vitepress'
import timeline from "vitepress-markdown-timeline";
// 只导入 withSidebar，不再需要 generateSidebar
import { withSidebar } from 'vitepress-sidebar';
import { withMermaid } from 'vitepress-plugin-mermaid';
import mathjax3 from 'markdown-it-mathjax3';

const REPO_NAME = '/Blog/'

// 你的 MathJax 自定义元素列表
const customElements = [
  'mjx-container', 'mjx-assistive-mml', 'math', 'maction', 'maligngroup', 'malignmark', 'menclose', 'merror', 'mfenced', 'mfrac', 'mi', 'mlongdiv', 'mmultiscripts', 'mn', 'mo', 'mover', 'mpadded', 'mphantom', 'mroot', 'mrow', 'ms', 'mscarries', 'mscarry', 'mscarries', 'msgroup', 'mstack', 'mlongdiv', 'msline', 'mstack', 'mspace', 'msqrt', 'msrow', 'mstack', 'mstack', 'mstyle', 'msub', 'msup', 'msubsup', 'mtable', 'mtd', 'mtext', 'mtr', 'munder', 'munderover', 'semantics', 'math', 'mi', 'mn', 'mo', 'ms', 'mspace', 'mtext', 'menclose', 'merror', 'mfenced', 'mfrac', 'mpadded', 'mphantom', 'mroot', 'mrow', 'msqrt', 'mstyle', 'mmultiscripts', 'mover', 'mprescripts', 'msub', 'msubsup', 'msup', 'munder', 'munderover', 'none', 'maligngroup', 'malignmark', 'mtable', 'mtd', 'mtr', 'mlongdiv', 'mscarries', 'mscarry', 'msgroup', 'msline', 'msrow', 'mstack', 'maction', 'semantics', 'annotation', 'annotation-xml',
];

// https://vitepress.dev/reference/site-config
// 基础配置，移除了 sidebar 属性，因为它将由 withSidebar 自动生成
const baseVitePressConfig = {
  base: REPO_NAME,
  title: "下午好",
  description: "来杯下午茶",
  head: [
    ['link', { rel: 'icon', href: `${REPO_NAME}/header1.png` }]
  ],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    siteTitle: "下午茶,渍",
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: "烧烤",
            buttonAriaLabel: "烧烤中ing",
          },
          modal: {
            noResultsText: "没有找到结果",
            resetButtonTitle: "清除搜索条件",
            footer: {
              selectText: "选择",
              navigateText: "切换",
              closeText: "关闭",
            },
          }
        }
      }
    },
    nav: [
      { text: '主页', link: '/' },
      { text: 'frontend', link: '/frontend/1.preface' }, // 保持与 sidebar 里的 /frontend/ 一致
      { text: 'llm', link: '/llm/1.preface' },           // 保持与 sidebar 里的 /llm/ 一致
      { text: 'linux', link: '/linux/1.preface' },       // 纠正了拼写，并与 /linux/ 一致
      { text: 'agent', link: '/agent/1.preface' },       // 保持与 sidebar 里的 /agent/ 一致
      { text: 'tool', link: '/tool/1.preface' }      // 保持与 sidebar 里的 /tool/ 一致

    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/lishiyu2006' }
    ],
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag),
      },
    },
  },
  markdown: {
    config: (md) => {
      // 在这里全局应用 markdown-it 插件
      md.use(timeline);
      md.use(mathjax3);

      // 你的自定义 heading_close 规则，并保留原始规则
      const originalHeadingCloseRule = md.renderer.rules.heading_close;
      md.renderer.rules.heading_close = (tokens, idx, options, env, slf) => {
        let htmlResult = originalHeadingCloseRule
            ? originalHeadingCloseRule(tokens, idx, options, env, slf)
            : slf.renderToken(tokens, idx, options);
        if (tokens[idx].tag === 'h1') htmlResult += `<ArticleMetadata />`;
        return htmlResult;
      };
    }
  },
  // 移除了这里的 sidebar: generateSidebar(...)，完全交给 withSidebar 处理
};

// 你的侧边栏扫描配置数组
// 建议：将 documentRootPath 设置为 '/'，代表 VitePress 的源目录
const sidebarConfigs = [
  {
    documentRootPath: '/',       // 1. 你的项目文档根目录（通常是 docs 或项目根目录）
    scanStartPath: 'frontend',       // 2. 告诉插件：去扫描项目下的 “前端” 这个文件夹
    basePath: '/frontend/',      // 3. 路由前缀。把“前端/xxx.md”变成“/frontend/xxx.html”
    resolvePath: '/frontend/',  // 4. 侧边栏激活路径。当浏览器地址是 /frontend/ 开头时，显示这个侧边栏
    rootGroupText: '现在',       // 5. 这个分组在侧边栏最顶层显示的名称叫“现在”

    removePrefixAfterOrdering: true, // 6. 如果你的文件名叫 "1.preface.md"，在侧边栏显示时会自动去掉前面的数字，变成 "preface"
    prefixSeparator: '.',            // 7. 指定序号的分隔符（配合上面那个用，这里是点 `.`）

    collapsed: true,      // 8. 侧边栏默认折叠
    collapseDepth: 2,     // 9. 超过 2 层的目录才折叠
  },
  {
    documentRootPath: '/',
    scanStartPath: 'linux',
    basePath: '/linux/',
    resolvePath: '/linux/',
    rootGroupText: 'linux',
    removePrefixAfterOrdering: true,
    prefixSeparator: '.',
    collapsed: true,
    collapseDepth: 2,
  },
  {
    documentRootPath: '/', // 使用 '/' 表示文档根目录
    scanStartPath: 'llm',
    basePath: '/llm/',
    resolvePath: '/llm/',
    rootGroupText: '过往',
    removePrefixAfterOrdering: true,
    prefixSeparator: '.',
    collapsed: true,
    collapseDepth: 2,
  },
  {
    documentRootPath: '/', // 使用 '/' 表示文档根目录
    scanStartPath: 'agent',
    basePath: '/agent/',
    resolvePath: '/agent/',
    rootGroupText: 'agent',
    removePrefixAfterOrdering: true,
    prefixSeparator: '.',
    collapsed: true,
    collapseDepth: 2,
  },
  {
    documentRootPath: '/',
    scanStartPath: 'tool',
    basePath: '/tool/',
    resolvePath: '/tool/',
    rootGroupText: 'tool',
    removePrefixAfterOrdering: true,
    prefixSeparator: '.',
    collapsed: true,
    collapseDepth: 2,
  }
];

// 1. 使用 withSidebar 包装基础配置，生成包含侧边栏的配置对象
const configWithSidebar = withSidebar(baseVitePressConfig, sidebarConfigs);

// 2. 使用 withMermaid 包装上一步的结果，并添加 mermaid 特定的配置
// 这种结构可以确保所有插件正确地包装和修改配置
export default defineConfig(
    withMermaid({
      // 将已经包含侧边栏的配置对象展开
      ...configWithSidebar,

      // 添加 mermaid 特定的配置
      mermaid: {
        // refer https://mermaid.js.org/config/setup/modules/mermaidAPI.html#mermaidapi-configuration-defaults for options
        // 你的 Mermaid 配置...
      },
      mermaidPlugin: {
        class: "mermaid my-class", // set additional css classes for parent container
        // 你的 Mermaid 插件配置...
      },
    })
);
