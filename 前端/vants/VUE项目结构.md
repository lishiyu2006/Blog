# VUE项目结构

## 默认情况

使用流行的前端框架搭建新项目时，组件结构是扁平的，完全不遵循任何层级结构
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202604281530525.png)
- **assets**目录存储静态资产，如图像、字体和应用中使用的 CSS 文件
- **components**目录包含可重用的 Vue 组件。建议采用扁平层级结构
- **main.js**文件作为应用的入口，支持 Vue 的初始化和插件或额外库的配置。

## 在应用中

任何大型应用的界面上可以看见的部分都应该拆分成不不同的 components部分

识别它们并不总是简单直接，但随着时间和经验积累会越来越好。

- **components**所有在整个应用程序中使用的共享组件。
- composables** ：所有共享的可组合。
- **config**: 应用程序配置文件。
- **features**: 包含所有应用功能。我们希望大部分应用代码都保留在这里。稍后会详细说明。
- **layouts** ：页面的不同版面。
- **lib**：适用于我们应用中使用的不同第三方库配置。
- **pages**：我们申请表的页面。
- **services**： 共享应用服务和提供者。
- **stores**：全球州级门店。
- **test**：与测试相关的模拟、辅助工具、工具和配置。
- **types**：共享 TypeScript 类型定义。
- **utils**：共享效用函数。

![Uploading file...avcte]()
