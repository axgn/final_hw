<template>
  <section class="comments">
    <h2 class="comments-title">评论区</h2>

    <div v-if="listLoading" class="comments-status">正在加载评论...</div>
    <div v-else-if="loadError" class="comments-error">{{ loadError }}</div>

    <ul v-else-if="comments.length" class="comments-list">
      <li v-for="c in comments" :key="c.id" class="comment-item">
        <div class="comment-header">
          <div class="comment-user">
            <img
              v-if="c.avatar_url"
              :src="c.avatar_url"
              alt="avatar"
              class="comment-avatar"
            />
            <div class="comment-meta">
              <span class="comment-username">{{ c.username || '匿名用户' }}</span>
              <span class="comment-time">{{ c.created_at }}</span>
            </div>
          </div>
          <button
            class="comment-delete"
            type="button"
            @click="handleDelete(c.id)"
            :disabled="deletingId === c.id || deletingId !== null || !hasToken"
          >
            删除
          </button>
        </div>
        <p class="comment-content">{{ c.content }}</p>
      </li>
    </ul>
    <p v-else class="comments-empty">还没有评论，快来抢沙发吧～</p>

    <div class="comments-divider"></div>

    <div class="comment-form">
      <h3 class="comment-form-title">发表评论</h3>
      <div v-if="!hasToken" class="comment-token-hint">
        <span>登录后即可发表评论。</span>
        <button type="button" class="comment-login-btn" @click="openAuthModal('login')">
          登录 / 注册
        </button>
      </div>
      <textarea
        v-model="newContent"
        class="comment-input"
        placeholder="说点什么吧..."
      ></textarea>
      <div class="comment-actions">
        <button
          type="button"
          class="comment-submit"
          @click="handleSubmit"
          :disabled="submitting || !trimmedContent"
        >
          {{ submitting ? '提交中...' : '发表评论' }}
        </button>
        <span v-if="submitError" class="comment-submit-error">{{ submitError }}</span>
        <span v-else-if="submitSuccess" class="comment-submit-success">评论成功</span>
      </div>
    </div>

    <div v-if="showAuthModal" class="auth-modal-backdrop">
      <div class="auth-modal" @click.stop>
        <div class="auth-modal-header">
          <h3 class="auth-modal-title">登录 / 注册</h3>
          <button type="button" class="auth-modal-close" @click="closeAuthModal">×</button>
        </div>

        <div class="auth-tabs">
          <button
            type="button"
            class="auth-tab"
            :class="{ active: authActiveTab === 'login' }"
            @click="authActiveTab = 'login'"
          >
            登录
          </button>
          <button
            type="button"
            class="auth-tab"
            :class="{ active: authActiveTab === 'register' }"
            @click="authActiveTab = 'register'"
          >
            注册
          </button>
        </div>

        <form v-if="authActiveTab === 'login'" class="auth-form" @submit.prevent="handleLogin">
          <div class="auth-field">
            <label>用户名</label>
            <input v-model="loginUsername" class="auth-input" placeholder="请输入用户名" />
          </div>
          <div class="auth-field">
            <label>密码</label>
            <input
              v-model="loginPassword"
              type="password"
              class="auth-input"
              placeholder="请输入密码"
            />
          </div>
          <button
            type="submit"
            class="auth-btn full"
            :disabled="loginLoading || !loginUsername.trim() || !loginPassword.trim()"
          >
            {{ loginLoading ? '登录中...' : '登录' }}
          </button>
          <p v-if="loginError" class="auth-error">{{ loginError }}</p>
        </form>

        <form v-else class="auth-form" @submit.prevent="handleRegister">
          <div class="auth-field">
            <label>用户名</label>
            <input v-model="regUsername" class="auth-input" placeholder="请输入用户名" />
          </div>
          <div class="auth-field">
            <label>密码</label>
            <input
              v-model="regPassword"
              type="password"
              class="auth-input"
              placeholder="请输入密码"
            />
          </div>
          <div class="auth-field">
            <label>邮箱（可选）</label>
            <input
              v-model="regEmail"
              type="email"
              class="auth-input"
              placeholder="example@example.com"
            />
          </div>
          <button
            type="submit"
            class="auth-btn full"
            :disabled="regLoading || !regUsername.trim() || !regPassword.trim()"
          >
            {{ regLoading ? '注册中...' : '注册' }}
          </button>
          <p v-if="regError" class="auth-error">{{ regError }}</p>
          <p v-if="regSuccess" class="auth-success">注册成功，请使用该账号登录。</p>
        </form>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'

const props = defineProps({
  postId: {
    type: Number,
    required: true,
  },
})

const COMMENTS_API_BASE = '/api/comments'
const USER_API_BASE = '/api/user'

const comments = ref([])
const listLoading = ref(false)
const loadError = ref('')

const newContent = ref('')
const submitting = ref(false)
const submitError = ref('')
const submitSuccess = ref(false)

const deletingId = ref(null)

const authToken = ref(localStorage.getItem('auth_token') || '')

const trimmedContent = computed(() => newContent.value.trim())

const hasToken = computed(() => !!authToken.value && authToken.value.trim().length > 0)

const showAuthModal = ref(false)
const authActiveTab = ref('login')

const loginUsername = ref('')
const loginPassword = ref('')
const loginLoading = ref(false)
const loginError = ref('')

const regUsername = ref('')
const regPassword = ref('')
const regEmail = ref('')
const regLoading = ref(false)
const regError = ref('')
const regSuccess = ref(false)

function syncAuthToken() {
  authToken.value = localStorage.getItem('auth_token') || ''
}

function setToken(token) {
  if (token) {
    localStorage.setItem('auth_token', token)
  } else {
    localStorage.removeItem('auth_token')
  }
  window.dispatchEvent(new CustomEvent('auth-changed'))
}

function getAuthHeaders() {
  if (!authToken.value) return {}
  return {
    Authorization: `Bearer ${authToken.value}`,
  }
}

function openAuthModal(tab = 'login') {
  authActiveTab.value = tab
  loginError.value = ''
  regError.value = ''
  regSuccess.value = false
  showAuthModal.value = true
}

function closeAuthModal() {
  showAuthModal.value = false
}

async function loadComments() {
  listLoading.value = true
  loadError.value = ''
  try {
    const resp = await fetch(`${COMMENTS_API_BASE}?post_id=${encodeURIComponent(props.postId)}`)
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`)
    }
    const body = await resp.json()
    if (body.code !== 0) {
      throw new Error(body.message || '加载评论失败')
    }
    comments.value = Array.isArray(body.data) ? body.data : []
  } catch (e) {
    console.error(e)
    loadError.value = '加载评论失败，请稍后重试。'
  } finally {
    listLoading.value = false
  }
}

async function handleSubmit() {
  submitError.value = ''
  submitSuccess.value = false

  if (!trimmedContent.value) return
  if (!hasToken.value) {
    openAuthModal('login')
    return
  }

  submitting.value = true
  try {
    const resp = await fetch(COMMENTS_API_BASE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify({
        post_id: props.postId,
        content: trimmedContent.value,
      }),
    })

    const body = await resp.json().catch(() => ({ code: 1, message: '未知错误' }))

    if (!resp.ok || body.code !== 0) {
      const msg = body.message || `发表评论失败 (HTTP ${resp.status})`
      submitError.value = msg
      return
    }

    submitSuccess.value = true
    newContent.value = ''
    await loadComments()
  } catch (e) {
    console.error(e)
    submitError.value = '发表评论失败，请稍后重试。'
  } finally {
    submitting.value = false
    setTimeout(() => {
      submitSuccess.value = false
    }, 1500)
  }
}

async function handleDelete(id) {
  if (!hasToken.value) {
    submitError.value = '请先登录后再删除评论。'
    openAuthModal('login')
    return
  }

  deletingId.value = id
  submitError.value = ''

  try {
    const resp = await fetch(`${COMMENTS_API_BASE}/${id}`, {
      method: 'DELETE',
      headers: {
        ...getAuthHeaders(),
      },
    })

    const body = await resp.json().catch(() => ({ code: 1, message: '未知错误' }))

    if (resp.status === 404 || body.code === 1) {
      submitError.value = body.message || '评论不存在或无权限删除。'
    } else if (!resp.ok || body.code !== 0) {
      submitError.value = body.message || `删除失败 (HTTP ${resp.status})`
    } else {
      await loadComments()
    }
  } catch (e) {
    console.error(e)
    submitError.value = '删除评论失败，请稍后重试。'
  } finally {
    deletingId.value = null
  }
}

async function handleLogin() {
  loginError.value = ''
  regSuccess.value = false

  if (!loginUsername.value.trim() || !loginPassword.value.trim()) return

  loginLoading.value = true
  try {
    const resp = await fetch(`${USER_API_BASE}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: loginUsername.value.trim(),
        password: loginPassword.value.trim(),
      }),
    })

    const body = await resp.json().catch(() => ({ code: 1, message: '未知错误' }))

    if (!resp.ok || body.code !== 0) {
      loginError.value = body.message || `登录失败 (HTTP ${resp.status})`
      return
    }

    if (body.token) {
      setToken(body.token)
      syncAuthToken()
    }

    closeAuthModal()
  } catch (e) {
    console.error(e)
    loginError.value = '登录失败，请稍后重试。'
  } finally {
    loginLoading.value = false
  }
}

async function handleRegister() {
  regError.value = ''
  regSuccess.value = false

  if (!regUsername.value.trim() || !regPassword.value.trim()) return

  regLoading.value = true
  try {
    const resp = await fetch(`${USER_API_BASE}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: regUsername.value.trim(),
        password: regPassword.value.trim(),
        email: regEmail.value.trim() || undefined,
      }),
    })

    const body = await resp.json().catch(() => ({ code: 1, message: '未知错误' }))

    if (!resp.ok || body.code !== 0) {
      regError.value = body.message || `注册失败 (HTTP ${resp.status})`
      return
    }

    regSuccess.value = true
    authActiveTab.value = 'login'
  } catch (e) {
    console.error(e)
    regError.value = '注册失败，请稍后重试。'
  } finally {
    regLoading.value = false
  }
}

onMounted(() => {
  syncAuthToken()
  window.addEventListener('auth-changed', syncAuthToken)
  loadComments()
})

watch(
  () => props.postId,
  () => {
    comments.value = []
    loadComments()
  },
)
</script>

<style scoped>
.comments {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e5e7eb;
  max-width: 760px;
}

.comments-title {
  margin: 0 0 0.75rem;
  font-size: 1.1rem;
  color: #111827;
}

.comments-status {
  color: #6b7280;
  font-size: 0.9rem;
}

.comments-error {
  color: #b91c1c;
  font-size: 0.9rem;
}

.comments-empty {
  color: #6b7280;
  font-size: 0.9rem;
}

.comments-list {
  list-style: none;
  padding: 0;
  margin: 0 0 1rem;
}

.comment-item {
  padding: 0.75rem 0;
  border-bottom: 1px solid #e5e7eb;
}

.comment-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.25rem;
}

.comment-user {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.comment-avatar {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  object-fit: cover;
}

.comment-meta {
  display: flex;
  flex-direction: column;
}

.comment-username {
  font-size: 0.95rem;
  color: #111827;
}

.comment-time {
  font-size: 0.8rem;
  color: #9ca3af;
}

.comment-content {
  margin: 0.25rem 0 0;
  font-size: 0.95rem;
  color: #111827;
}

.comment-delete {
  border: none;
  background: transparent;
  color: #ef4444;
  font-size: 0.8rem;
  cursor: pointer;
}

.comment-delete:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.comments-divider {
  margin: 1.25rem 0 1rem;
  border-top: 1px dashed #e5e7eb;
}

.comment-form-title {
  margin: 0 0 0.5rem;
  font-size: 1rem;
  color: #111827;
}

.comment-token-hint {
  margin: 0 0 0.5rem;
  font-size: 0.8rem;
  color: #6b7280;
}

.comment-token-hint code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    'Liberation Mono', 'Courier New', monospace;
  background: #f3f4f6;
  padding: 0.05rem 0.25rem;
  border-radius: 3px;
}

.comment-input {
  width: 100%;
  min-height: 80px;
  padding: 0.6rem 0.8rem;
  border-radius: 6px;
  border: 1px solid #d1d5db;
  font-family: inherit;
  resize: vertical;
}

.comment-actions {
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.comment-submit {
  padding: 0.35rem 1rem;
  border-radius: 999px;
  border: none;
  background: #111827;
  color: #f9fafb;
  cursor: pointer;
  font-size: 0.9rem;
}

.comment-submit:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.comment-submit-error {
  font-size: 0.85rem;
  color: #b91c1c;
}

.comment-submit-success {
  font-size: 0.85rem;
  color: #16a34a;
}

.comment-login-btn {
  margin-left: 0.5rem;
  padding: 0.1rem 0.6rem;
  border-radius: 999px;
  border: none;
  background: #111827;
  color: #f9fafb;
  font-size: 0.75rem;
  cursor: pointer;
}

.auth-modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 30;
}

.auth-modal {
  width: 320px;
  max-width: 90vw;
  background: #ffffff;
  border-radius: 0.75rem;
  padding: 1rem 1.25rem 1rem;
  box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.2),
    0 10px 10px -5px rgba(15, 23, 42, 0.1);
}

.auth-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.auth-modal-title {
  margin: 0;
  font-size: 1rem;
  color: #111827;
}

.auth-modal-close {
  border: none;
  background: transparent;
  font-size: 1.1rem;
  line-height: 1;
  cursor: pointer;
  color: #6b7280;
}

.auth-tabs {
  display: flex;
  margin-bottom: 0.5rem;
  border-radius: 999px;
  background: #f3f4f6;
  padding: 2px;
}

.auth-tab {
  flex: 1;
  border: none;
  background: transparent;
  padding: 0.25rem 0.5rem;
  border-radius: 999px;
  font-size: 0.85rem;
  cursor: pointer;
  color: #6b7280;
}

.auth-tab.active {
  background: #111827;
  color: #f9fafb;
}

.auth-form {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.auth-field {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.auth-field label {
  font-size: 0.8rem;
  color: #4b5563;
}

.auth-input {
  padding: 0.35rem 0.5rem;
  border-radius: 0.375rem;
  border: 1px solid #d1d5db;
  font-size: 0.85rem;
}

.auth-btn {
  border: none;
  border-radius: 999px;
  padding: 0.3rem 0.9rem;
  font-size: 0.85rem;
  cursor: pointer;
  background: #111827;
  color: #f9fafb;
}

.auth-btn.full {
  width: 100%;
  justify-content: center;
}

.auth-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.auth-error {
  margin-top: 0.2rem;
  font-size: 0.8rem;
  color: #b91c1c;
}

.auth-success {
  margin-top: 0.2rem;
  font-size: 0.8rem;
  color: #16a34a;
}
</style>
