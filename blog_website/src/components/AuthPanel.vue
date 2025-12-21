<template>
  <section class="auth">
    <h2 class="auth-title">用户中心</h2>

    <div v-if="profileLoading" class="auth-status">正在加载用户信息...</div>
    <p v-else-if="profileError" class="auth-error">{{ profileError }}</p>

    <div v-if="currentUser" class="auth-profile">
      <div class="auth-profile-main">
        <img
          v-if="currentUser.avatar_url"
          :src="currentUser.avatar_url"
          alt="avatar"
          class="auth-avatar"
        />
        <div class="auth-profile-text">
          <div class="auth-username">{{ currentUser.username }}</div>
          <div class="auth-email" v-if="currentUser.email">{{ currentUser.email }}</div>
        </div>
      </div>
      <div class="auth-profile-actions">
        <button type="button" class="auth-btn secondary" @click="refreshProfile">
          刷新资料
        </button>
        <button type="button" class="auth-btn" @click="logout">退出登录</button>
      </div>
    </div>

    <div v-else class="auth-forms">
      <div class="auth-tabs">
        <button
          type="button"
          class="auth-tab"
          :class="{ active: activeTab === 'login' }"
          @click="activeTab = 'login'"
        >
          登录
        </button>
        <button
          type="button"
          class="auth-tab"
          :class="{ active: activeTab === 'register' }"
          @click="activeTab = 'register'"
        >
          注册
        </button>
      </div>

      <form v-if="activeTab === 'login'" class="auth-form" @submit.prevent="handleLogin">
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

      <form
        v-else
        class="auth-form"
        @submit.prevent="handleRegister"
      >
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
  </section>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'

const USER_API_BASE = '/api/user'

const state = reactive({
  currentUser: null,
})

const profileLoading = ref(false)
const profileError = ref('')

const activeTab = ref('login')

// 登录表单
const loginUsername = ref('')
const loginPassword = ref('')
const loginLoading = ref(false)
const loginError = ref('')

// 注册表单
const regUsername = ref('')
const regPassword = ref('')
const regEmail = ref('')
const regLoading = ref(false)
const regError = ref('')
const regSuccess = ref(false)

const currentUser = ref(null)

function getToken() {
  return localStorage.getItem('auth_token') || ''
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
  const token = getToken()
  if (!token) return {}
  return {
    Authorization: `Bearer ${token}`,
  }
}

async function refreshProfile() {
  const token = getToken()
  if (!token) {
    currentUser.value = null
    return
  }

  profileLoading.value = true
  profileError.value = ''

  try {
    const resp = await fetch(`${USER_API_BASE}/profile`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    const body = await resp.json().catch(() => ({ code: 1, message: '未知错误' }))

    if (!resp.ok || body.code !== 0) {
      if (resp.status === 401) {
        // token 无效，清理本地
        setToken('')
        currentUser.value = null
      }
      profileError.value = body.message || `获取用户信息失败 (HTTP ${resp.status})`
      return
    }

    currentUser.value = body.data || null
  } catch (e) {
    console.error(e)
    profileError.value = '获取用户信息失败，请稍后重试。'
  } finally {
    profileLoading.value = false
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
    }

    currentUser.value = body.data || null
    loginPassword.value = ''
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
    activeTab.value = 'login'
  } catch (e) {
    console.error(e)
    regError.value = '注册失败，请稍后重试。'
  } finally {
    regLoading.value = false
  }
}

function logout() {
  setToken('')
  currentUser.value = null
}

onMounted(() => {
  if (getToken()) {
    refreshProfile()
  }
})
</script>

<style scoped>
.auth {
  margin-bottom: 1.5rem;
  padding: 1rem;
  border-radius: 0.5rem;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.auth-title {
  margin: 0 0 0.5rem;
  font-size: 1rem;
  color: #111827;
}

.auth-status {
  font-size: 0.85rem;
  color: #6b7280;
}

.auth-error {
  margin-top: 0.4rem;
  font-size: 0.85rem;
  color: #b91c1c;
}

.auth-success {
  margin-top: 0.4rem;
  font-size: 0.85rem;
  color: #16a34a;
}

.auth-profile {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.auth-profile-main {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.auth-avatar {
  width: 36px;
  height: 36px;
  border-radius: 999px;
  object-fit: cover;
}

.auth-profile-text {
  display: flex;
  flex-direction: column;
}

.auth-username {
  font-size: 0.95rem;
  color: #111827;
}

.auth-email {
  font-size: 0.8rem;
  color: #6b7280;
}

.auth-profile-actions {
  display: flex;
  gap: 0.5rem;
}

.auth-forms {
  margin-top: 0.25rem;
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

.auth-btn.secondary {
  background: #e5e7eb;
  color: #111827;
}

.auth-btn.full {
  width: 100%;
  justify-content: center;
}

.auth-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
