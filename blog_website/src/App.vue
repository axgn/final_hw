<template>
  <div class="app">
    <header class="app-header">
      <h1>我的 Markdown 博客</h1>
    </header>

    <main class="app-main">
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
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import MarkdownViewer from './components/MarkdownViewer.vue'
import posts from './posts'

const currentSlug = ref(posts[0]?.slug || '')

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
}

.app-header {
  padding: 1rem 2rem;
  background: #1e293b;
  color: #f9fafb;
}

.app-main {
  display: grid;
  grid-template-columns: 240px 1fr;
  flex: 1;
  min-height: 0;
}

.sidebar {
  border-right: 1px solid #e5e7eb;
  padding: 1rem;
  background: #f9fafb;
}

.sidebar h2 {
  margin-top: 0;
  font-size: 1rem;
  color: #4b5563;
}

.sidebar ul {
  list-style: none;
  padding: 0;
  margin: 0.5rem 0 0;
}

.sidebar li {
  padding: 0.3rem 0.2rem;
  cursor: pointer;
  border-radius: 4px;
}

.sidebar li:hover {
  background: #e5e7eb;
}

.sidebar li.active {
  background: #111827;
  color: #f9fafb;
}

.content {
  padding: 1.5rem 2rem;
  overflow: auto;
}

.sentiment {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e5e7eb;
}

.sentiment h2 {
  margin-top: 0;
  font-size: 1.1rem;
  color: #111827;
}

.sentiment-input {
  width: 100%;
  min-height: 100px;
  padding: 0.6rem 0.8rem;
  border-radius: 6px;
  border: 1px solid #d1d5db;
  font-family: inherit;
  resize: vertical;
}

.sentiment-actions {
  margin-top: 0.75rem;
}

.sentiment-actions button {
  padding: 0.4rem 1rem;
  border-radius: 999px;
  border: none;
  background: #111827;
  color: #f9fafb;
  cursor: pointer;
}

.sentiment-actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.sentiment-error {
  margin-top: 0.5rem;
  color: #b91c1c;
}

.sentiment-result {
  margin-top: 0.75rem;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  background: #f3f4f6;
}
</style>
