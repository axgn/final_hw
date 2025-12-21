<template>
  <section class="video-page">
    <header class="video-header">
      <h2>视频播放</h2>
      <p class="video-subtitle">在这里可以播放你自己上传或托管的视频文件。</p>
    </header>

    <div class="video-player-card">
      <video
        class="video-element"
        controls
        :src="videoSrc"
        poster="/images/homework/video_poster.png"
      >
        您的浏览器不支持 video 标签。
      </video>
      <p class="video-tip">
        当前示例视频地址为 <code>/videos/demo.mp4</code>，你可以将视频文件放在
        <code>public/videos</code> 目录下并修改这里的路径。
      </p>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

// 这里可以根据需要改成从配置或后端获取
const videoSrc = computed(() => '/videos/demo.mp4')
</script>

<style scoped>
.video-page {
  max-width: 860px;
  margin: 0 auto;
  color: #e5e7eb;
}

.video-header h2 {
  margin: 0 0 0.4rem;
  font-size: 1.3rem;
  font-weight: 600;
}

.video-subtitle {
  margin: 0;
  font-size: 0.9rem;
  color: #9ca3af;
}

.video-player-card {
  margin-top: 1.25rem;
  padding: 1rem 1.1rem 1.15rem;
  border-radius: 0.9rem;
  background: radial-gradient(circle at top, #020617 0, #020617 45%, #030712 100%);
  box-shadow: 0 24px 40px -24px rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(55, 65, 81, 0.9);
}

.video-element {
  width: 100%;
  max-height: 480px;
  border-radius: 0.75rem;
  outline: none;
  background: #000;
}

.video-tip {
  margin-top: 0.7rem;
  font-size: 0.85rem;
  color: #9ca3af;
}

.video-tip code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    'Liberation Mono', 'Courier New', monospace;
  background: rgba(15, 23, 42, 0.9);
  padding: 0.05rem 0.25rem;
  border-radius: 3px;
}
</style>
