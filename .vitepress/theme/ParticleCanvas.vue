<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const canvasRef = ref<HTMLCanvasElement | null>(null)

let animId = 0
let mouse = { x: -9999, y: -9999 }

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  opacity: number
}

function isDarkMode(): boolean {
  return document.documentElement.classList.contains('dark')
}

function getParticleColor(dark: boolean): string {
  if (dark) {
    // 暗色主题：白色 / 浅灰系
    const lightColors = [
      'rgba(255,255,255,1)',
      'rgba(220,220,255,1)',
      'rgba(200,230,255,1)',
      'rgba(240,220,255,1)',
      'rgba(255,240,220,1)',
    ]
    return lightColors[Math.floor(Math.random() * lightColors.length)]
  } else {
    // 亮色主题：深色系
    const darkColors = [
      'rgba(80,80,120,1)',
      'rgba(60,100,160,1)',
      'rgba(100,60,140,1)',
      'rgba(50,120,100,1)',
      'rgba(120,80,60,1)',
    ]
    return darkColors[Math.floor(Math.random() * darkColors.length)]
  }
}

function getLineColor(dark: boolean): string {
  return dark
    ? 'rgba(200, 210, 255,'
    : 'rgba(80, 100, 160,'
}

function createParticle(w: number, h: number, dark: boolean): Particle & { color: string } {
  const angle = Math.random() * Math.PI * 2
  const speed = 0.15 + Math.random() * 0.35
  return {
    x: Math.random() * w,
    y: Math.random() * h,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    radius: 1.8 + Math.random() * 2.2,
    opacity: 0.4 + Math.random() * 0.5,
    color: getParticleColor(dark),
  }
}

const cleanupFns: (() => void)[] = []

onMounted(() => {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')!

  let W = (canvas.width = window.innerWidth)
  let H = (canvas.height = window.innerHeight)
  let dark = isDarkMode()
  const COUNT = Math.min(100, Math.floor((W * H) / 14000))
  let particles = Array.from({ length: COUNT }, () => createParticle(W, H, dark))

  const CONNECT_DIST = 140
  const MOUSE_DIST = 170
  const ATTRACT = 0.013
  const MAX_SPEED = 0.8       // 最大速度限制
  const WANDER = 0.008        // 随机游走扰动强度

  // Watch theme changes
  const observer = new MutationObserver(() => {
    dark = isDarkMode()
    // Recolor existing particles
    for (const p of particles) {
      (p as any).color = getParticleColor(dark)
    }
  })
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })

  function onResize() {
    W = canvas.width = window.innerWidth
    H = canvas.height = window.innerHeight
    dark = isDarkMode()
    particles = Array.from({ length: COUNT }, () => createParticle(W, H, dark))
  }

  function onMouseMove(e: MouseEvent) {
    mouse.x = e.clientX
    mouse.y = e.clientY
  }

  function onMouseLeave() {
    mouse.x = -9999
    mouse.y = -9999
  }

  window.addEventListener('resize', onResize)
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseleave', onMouseLeave)

  function draw() {
    ctx.clearRect(0, 0, W, H)

    // --- update ---
    for (const p of particles) {
      // 1. 随机游走：每帧施加微小随机扰动，保证粒子持续运动
      p.vx += (Math.random() - 0.5) * WANDER
      p.vy += (Math.random() - 0.5) * WANDER

      // 2. 鼠标吸引力
      const dx = mouse.x - p.x
      const dy = mouse.y - p.y
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < MOUSE_DIST && dist > 0) {
        const force = (1 - dist / MOUSE_DIST) * ATTRACT
        p.vx += (dx / dist) * force
        p.vy += (dy / dist) * force
      }

      // 3. 限制最大速度，防止粒子飞得过快
      const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy)
      if (spd > MAX_SPEED) {
        p.vx = (p.vx / spd) * MAX_SPEED
        p.vy = (p.vy / spd) * MAX_SPEED
      }

      // 4. 轻微阻尼（比之前小，避免速度归零）
      p.vx *= 0.999
      p.vy *= 0.999

      p.x += p.vx
      p.y += p.vy

      // 边界循环
      if (p.x < 0) p.x = W
      if (p.x > W) p.x = 0
      if (p.y < 0) p.y = H
      if (p.y > H) p.y = 0
    }

    const lineBase = getLineColor(dark)

    // --- connections between particles ---
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i]
        const b = particles[j]
        const dx = a.x - b.x
        const dy = a.y - b.y
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d < CONNECT_DIST) {
          const alpha = (1 - d / CONNECT_DIST) * 0.5
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(b.x, b.y)
          ctx.strokeStyle = `${lineBase}${alpha})`
          ctx.lineWidth = 0.7
          ctx.stroke()
        }
      }
    }

    // --- connections to mouse ---
    for (const p of particles) {
      const dx = mouse.x - p.x
      const dy = mouse.y - p.y
      const d = Math.sqrt(dx * dx + dy * dy)
      if (d < MOUSE_DIST) {
        const alpha = (1 - d / MOUSE_DIST) * 0.7
        ctx.beginPath()
        ctx.moveTo(p.x, p.y)
        ctx.lineTo(mouse.x, mouse.y)
        ctx.strokeStyle = `${lineBase}${alpha})`
        ctx.lineWidth = 0.9
        ctx.stroke()
      }
    }

    // --- draw particles ---
    for (const p of particles) {
      const color = (p as any).color as string
      // solid dot
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.globalAlpha = p.opacity
      ctx.fill()
      // soft glow halo
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 3)
      grad.addColorStop(0, color)
      grad.addColorStop(1, 'transparent')
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.radius * 3, 0, Math.PI * 2)
      ctx.fillStyle = grad
      ctx.globalAlpha = p.opacity * 0.35
      ctx.fill()
      ctx.globalAlpha = 1
    }

    animId = requestAnimationFrame(draw)
  }

  draw()

  cleanupFns.push(() => {
    cancelAnimationFrame(animId)
    observer.disconnect()
    window.removeEventListener('resize', onResize)
    window.removeEventListener('mousemove', onMouseMove)
    window.removeEventListener('mouseleave', onMouseLeave)
  })
})

onUnmounted(() => {
  cleanupFns.forEach(fn => fn())
  cleanupFns.length = 0
})
</script>

<template>
  <canvas ref="canvasRef" class="particle-canvas" />
</template>

<style scoped>
.particle-canvas {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
  z-index: 0;
}
</style>
