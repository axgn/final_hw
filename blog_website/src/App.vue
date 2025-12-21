<template>
  <div class="app">
    <header class="app-header">
      <h1>我的 Markdown 博客</h1>
      <nav class="app-nav">
        <button
          type="button"
          class="app-nav-btn"
          :class="{ active: activePage === 'blog' }"
          @click="activePage = 'blog'"
        >
          博客
        </button>
        <button
          type="button"
          class="app-nav-btn"
          :class="{ active: activePage === 'video' }"
          @click="activePage = 'video'"
        >
          视频
        </button>
      </nav>
    </header>

    <main v-if="activePage === 'blog'" class="app-main">
      <section class="sidebar">
        <h2>文章列表</h2>
        <ul>
          <li
            v-for="post in posts"
            :key="post.slug"
            :class="{ active: post.slug === currentSlug }"
            @click="selectPost(post.slug)"
          >
            {{ post.title }}
          </li>
        </ul>
      </section>

      <section class="content">
        <MarkdownViewer v-if="currentPost" :source="currentPost.content" />
        <p v-else>请选择一篇文章。</p>

        <CommentsSection v-if="currentPost" :post-id="currentPost.postId" />

        <section class="sentiment">
          <h2>情感分析小工具</h2>
          <textarea
            v-model="text"
            class="sentiment-input"
            placeholder="在这里输入一段中文句子，比如：这家店真的很好，强烈推荐！"
          ></textarea>
          <div class="sentiment-actions">
            <button @click="analyze" :disabled="loading || !text.trim()">
              {{ loading ? '分析中...' : '开始分析' }}
            </button>
          </div>
          <p v-if="error" class="sentiment-error">{{ error }}</p>
          <div v-if="result" class="sentiment-result">
            <p>预测结果：<strong>{{ result.label === 'pos' ? '正面' : '负面' }}</strong></p>
            <p>置信度：{{ (result.prob * 100).toFixed(2) }}%</p>
          </div>
        </section>
      </section>
    </main>

    <main v-else class="app-main single-column">
      <section class="content">
        <VideoPlayer />
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import MarkdownViewer from './components/MarkdownViewer.vue'
import CommentsSection from './components/CommentsSection.vue'
import VideoPlayer from './components/VideoPlayer.vue'
import posts from './posts'

const currentSlug = ref(posts[0]?.slug || '')

const activePage = ref('blog')

const currentPost = computed(() =>
  posts.find((p) => p.slug === currentSlug.value) || null,
)

function selectPost(slug) {
  currentSlug.value = slug
}

// ====== 调用 FastAPI 情感分析接口 ======
// 通过反向代理，将 /api/infer 转发到实际的 FastAPI 服务
const API_BASE_URL = '/api/infer'

const text = ref('')
const loading = ref(false)
const error = ref('')
const result = ref(null)

async function analyze() {
  error.value = ''
  result.value = null
  if (!text.value.trim()) return

  loading.value = true
  try {
    const resp = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: text.value }),
    })

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`)
    }

    const data = await resp.json()
    // 期望后端返回 { label: 'pos' | 'neg', prob: number }
    result.value = {
      label: data.label,
      prob: data.prob,
    }
  } catch (e) {
    error.value = '调用情感分析接口失败，请稍后重试。'
    console.error(e)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI',
    sans-serif;
  background: radial-gradient(circle at top left, #1e293b 0, #020617 50%, #000 100%);
  color: #e5e7eb;
  padding: 1.5rem 1.25rem 2rem;
  box-sizing: border-box;
}

.app-header {
  max-width: 1120px;
  margin: 0 auto 1rem;
  padding: 0.75rem 1.5rem;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.9);
  color: #e5e7eb;
  box-shadow: 0 25px 45px -18px rgba(15, 23, 42, 0.8);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.app-header h1 {
  font-size: 1.2rem;
  font-weight: 600;
  letter-spacing: 0.03em;
}

.app-nav {
  display: flex;
  gap: 0.5rem;
  background: rgba(15, 23, 42, 0.85);
  padding: 2px;
  border-radius: 999px;
}

.app-nav-btn {
  border: none;
  border-radius: 999px;
  padding: 0.25rem 0.9rem;
  font-size: 0.85rem;
  cursor: pointer;
  background: transparent;
  color: #9ca3af;
  transition: background 0.15s ease, color 0.15s ease, transform 0.12s ease;
}

.app-nav-btn.active {
  background: linear-gradient(90deg, #2563eb, #4f46e5);
  color: #f9fafb;
  transform: translateY(-0.5px);
}

.app-main {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  flex: 1;
  min-height: 0;
  max-width: 1120px;
  margin: 0 auto;
  border-radius: 1.25rem;
  background: rgba(15, 23, 42, 0.98);
  box-shadow: 0 30px 60px -24px rgba(15, 23, 42, 0.95);
  overflow: hidden;
}

.app-main.single-column {
  grid-template-columns: minmax(0, 1fr);
}

.sidebar {
  border-right: 1px solid rgba(55, 65, 81, 0.9);
  padding: 1.1rem 1rem;
  background: radial-gradient(circle at top, #020617 0, #020617 40%, #030712 100%);
}

.sidebar h2 {
  margin-top: 0;
  font-size: 0.95rem;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.sidebar ul {
  list-style: none;
  padding: 0;
  margin: 0.75rem 0 0;
}

.sidebar li {
  padding: 0.45rem 0.6rem;
  cursor: pointer;
  border-radius: 0.5rem;
  font-size: 0.9rem;
  color: #e5e7eb;
  transition: background 0.15s ease, color 0.15s ease, transform 0.12s ease;
}

.sidebar li:hover {
  background: rgba(30, 64, 175, 0.3);
  transform: translateX(2px);
}

.sidebar li.active {
  background: linear-gradient(90deg, #2563eb, #4f46e5);
  color: #f9fafb;
  box-shadow: 0 12px 20px -14px rgba(37, 99, 235, 0.9);
}

.content {
  padding: 1.5rem 2rem 1.75rem;
  overflow: auto;
  background: radial-gradient(circle at top left, #020617 0, #020617 25%, #020617 50%, #030712 100%);
}

.sentiment {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(55, 65, 81, 0.8);
}

.sentiment h2 {
  margin-top: 0;
  font-size: 1.05rem;
  color: #e5e7eb;
}

.sentiment-input {
  width: 100%;
  min-height: 100px;
  padding: 0.6rem 0.8rem;
  border-radius: 6px;
  border: 1px solid rgba(75, 85, 99, 0.8);
  font-family: inherit;
  resize: vertical;
  background: rgba(15, 23, 42, 0.9);
  color: #e5e7eb;
}

.sentiment-actions {
  margin-top: 0.75rem;
}

.sentiment-actions button {
  padding: 0.4rem 1rem;
  border-radius: 999px;
  border: none;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  color: #022c22;
  cursor: pointer;
  font-weight: 500;
  box-shadow: 0 14px 28px -18px rgba(22, 163, 74, 0.9);
  transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
}

.sentiment-actions button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  box-shadow: none;
}

.sentiment-actions button:not(:disabled):hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
  box-shadow: 0 18px 32px -20px rgba(22, 163, 74, 0.9);
}

.sentiment-error {
  margin-top: 0.5rem;
  color: #fecaca;
}

.sentiment-result {
  margin-top: 0.75rem;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(55, 65, 81, 0.8);
}
</style>
