import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import ParticleCanvas from './ParticleCanvas.vue'
import HomeHeroImage from './HomeHeroImage.vue'
import './custom.css'

export default {
  extends: DefaultTheme,

  Layout() {
    return h(DefaultTheme.Layout, null, {
      // 粒子画布注入到 layout 最底层（全局显示）
      'layout-top': () => h(ParticleCanvas),

      // 首页 hero image 插槽，替换为自定义圆形浮动头像
      'home-hero-image': () => h(HomeHeroImage),
    })
  },
}
