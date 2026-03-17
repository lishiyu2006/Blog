<script setup lang="ts">
import { useData, withBase } from 'vitepress'

const { frontmatter } = useData()

// withBase 自动补全 base 路径，解决 /Blog/ 前缀问题
function resolvedSrc(src: string) {
  return withBase(src)
}
</script>

<template>
  <div class="hero-image-wrapper">
    <!-- 背后圆形光晕层 -->
    <div class="glow-ring glow-ring-1" />
    <div class="glow-ring glow-ring-2" />
    <div class="glow-ring glow-ring-3" />
    <!-- 头像图片 -->
    <img
      v-if="frontmatter.hero?.image?.src"
      :src="resolvedSrc(frontmatter.hero.image.src)"
      :alt="frontmatter.hero.image.alt || ''"
      class="hero-avatar"
    />
  </div>
</template>

<style scoped>
.hero-image-wrapper {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 240px;
  height: 240px;
  animation: float-bob 3.8s ease-in-out infinite;
}

/* 浮动上下动画 */
@keyframes float-bob {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-16px); }
}

/* 头像圆形 */
.hero-avatar {
  position: relative;
  z-index: 2;
  width: 192px;
  height: 192px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow:
    0 0 0 3px rgba(167, 139, 250, 0.6),
    0 0 24px 6px rgba(125, 211, 252, 0.4),
    0 0 56px 14px rgba(167, 139, 250, 0.2);
}

/* 光晕圆环 */
.glow-ring {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
}

.glow-ring-1 {
  width: 210px;
  height: 210px;
  background: radial-gradient(
    circle,
    rgba(125, 211, 252, 0.18) 0%,
    rgba(167, 139, 250, 0.12) 55%,
    transparent 72%
  );
  animation: pulse-glow 3.8s ease-in-out infinite;
  z-index: 1;
}

.glow-ring-2 {
  width: 260px;
  height: 260px;
  background: radial-gradient(
    circle,
    transparent 38%,
    rgba(167, 139, 250, 0.10) 58%,
    rgba(249, 168, 212, 0.07) 74%,
    transparent 84%
  );
  animation: pulse-glow 3.8s ease-in-out infinite 0.5s;
  z-index: 0;
}

.glow-ring-3 {
  width: 310px;
  height: 310px;
  background: radial-gradient(
    circle,
    transparent 48%,
    rgba(125, 211, 252, 0.05) 68%,
    rgba(167, 139, 250, 0.03) 82%,
    transparent 90%
  );
  animation: pulse-glow 3.8s ease-in-out infinite 1s;
  z-index: 0;
}

@keyframes pulse-glow {
  0%, 100% { transform: scale(1);     opacity: 0.75; }
  50%       { transform: scale(1.10); opacity: 1;    }
}
</style>
