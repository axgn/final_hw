<template>
  <article class="markdown" v-html="rendered"></article>
</template>

<script setup>
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/atom-one-dark.css'

const props = defineProps({
  source: {
    type: String,
    required: true,
  },
})

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
})

md.set({
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        const result = hljs.highlight(code, {
          language: lang,
          ignoreIllegals: true,
        })
        return `<pre class="hljs"><code>${result.value}</code></pre>`
      } catch (e) {
        // fall through to plain text escape
      }
    }

    const escaped = md.utils.escapeHtml(code)
    return `<pre class="hljs"><code>${escaped}</code></pre>`
  },
})

function normalizeImagePaths(text) {
  if (!text) return ''

  return String(text).replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    (match, alt, url) => {
      const trimmed = String(url).trim()

      // 已经是绝对路径或 http(s) 链接 / 相对路径的，保持不变
      if (
        /^(https?:)?\/\//.test(trimmed) ||
        trimmed.startsWith('/') ||
        trimmed.startsWith('./') ||
        trimmed.startsWith('../')
      ) {
        return match
      }

      return `![${alt}](/images/${trimmed})`
    },
  )
}

const normalizedSource = computed(() => normalizeImagePaths(props.source))

const rendered = computed(() => md.render(normalizedSource.value))
</script>

<style scoped>
.markdown {
  max-width: 760px;
  line-height: 1.8;
  color: #111827;
  font-size: 0.95rem;
}

.markdown h1,
.markdown h2,
.markdown h3 {
  margin: 1.2rem 0 0.6rem;
  font-weight: 600;
}

.markdown p {
  margin: 0.6rem 0;
}
.markdown :not(pre) > code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    'Liberation Mono', 'Courier New', monospace;
  background: #f3f4f6;
  padding: 0.1rem 0.3rem;
  border-radius: 3px;
}

.markdown pre {
  margin: 1rem 0;
  border-radius: 0.6rem;
  overflow: hidden;
  box-shadow: 0 10px 25px -12px rgba(15, 23, 42, 0.6);
}

.markdown a {
  color: #2563eb;
}
</style>
