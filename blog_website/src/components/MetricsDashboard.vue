<template>
  <section class="metrics">
    <header class="metrics-header">
      <div>
        <h2>服务器性能监控</h2>
        <p class="metrics-subtitle">
          实时查看集群 CPU / 内存 / Nginx 请求等核心指标。
        </p>
      </div>
      <div class="metrics-controls">
        <label class="metrics-interval-label">
          刷新间隔：
          <select v-model.number="intervalSec">
            <option :value="10">10 秒</option>
            <option :value="30">30 秒</option>
            <option :value="60">60 秒</option>
          </select>
        </label>
        <button type="button" class="metrics-refresh" @click="fetchAll" :disabled="loading">
          {{ loading ? '刷新中...' : '立即刷新' }}
        </button>
      </div>
    </header>

    <div class="metrics-grid">
      <article class="metric-card">
        <h3>集群 CPU 利用率</h3>
        <p class="metric-value">
          <span v-if="cpuUsage != null">{{ (cpuUsage * 100).toFixed(1) }}%</span>
          <span v-else>--</span>
        </p>
        <p class="metric-desc">基于 Prometheus 采集的 node / pod CPU 指标。</p>
      </article>

      <article class="metric-card">
        <h3>集群内存利用率</h3>
        <p class="metric-value">
          <span v-if="memUsage != null">{{ (memUsage * 100).toFixed(1) }}%</span>
          <span v-else>--</span>
        </p>
        <p class="metric-desc">总内存使用量 / 总可用内存。</p>
      </article>

      <article class="metric-card">
        <h3>Nginx QPS</h3>
        <p class="metric-value">
          <span v-if="nginxQps != null">{{ nginxQps.toFixed(1) }}</span>
          <span v-else>--</span>
        </p>
        <p class="metric-desc">最近几分钟内平均每秒请求数。</p>
      </article>
    </div>

    <p v-if="error" class="metrics-error">{{ error }}</p>
  </section>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'

// 通过环境变量配置 Prometheus HTTP API 网关地址
// 建议在集群中通过 Nginx 或 Ingress 暴露一个只读的 Prometheus 代理，
// 然后在构建时设置 VITE_METRICS_API_BASE，例如：/prometheus
const METRICS_BASE = import.meta.env.VITE_METRICS_API_BASE || '/prometheus'

const loading = ref(false)
const error = ref('')

const cpuUsage = ref(null)
const memUsage = ref(null)
const nginxQps = ref(null)

const intervalSec = ref(30)
let timerId = null

async function queryPrometheus(query) {
  const url = `${METRICS_BASE}/api/v1/query?query=${encodeURIComponent(query)}`
  const resp = await fetch(url)
  if (!resp.ok) {
    throw new Error(`Prometheus HTTP ${resp.status}`)
  }
  const data = await resp.json()
  if (data.status !== 'success') {
    throw new Error(data.error || 'Prometheus query failed')
  }
  return data.data
}

async function fetchCpuUsage() {
  // 示例：最近 5 分钟内的整体 CPU 使用率
  const q = 'sum(rate(container_cpu_usage_seconds_total{job="kubelet", image!=""}[5m])) / sum(machine_cpu_cores)'
  const result = await queryPrometheus(q)
  const value = result.result?.[0]?.value?.[1]
  cpuUsage.value = value != null ? Number(value) : null
}

async function fetchMemUsage() {
  // 示例：整体内存使用率
  const used = 'sum(container_memory_working_set_bytes{job="kubelet", image!=""})'
  const total = 'sum(machine_memory_bytes)'

  const [usedData, totalData] = await Promise.all([
    queryPrometheus(used),
    queryPrometheus(total),
  ])

  const usedVal = usedData.result?.[0]?.value?.[1]
  const totalVal = totalData.result?.[0]?.value?.[1]

  if (usedVal != null && totalVal != null && Number(totalVal) > 0) {
    memUsage.value = Number(usedVal) / Number(totalVal)
  } else {
    memUsage.value = null
  }
}

async function fetchNginxQps() {
  // 示例：基于 nginx_exporter 的请求速率（请根据实际 label 调整 job/instance）
  const q = 'sum(rate(nginx_http_requests_total[5m]))'
  const result = await queryPrometheus(q)
  const value = result.result?.[0]?.value?.[1]
  nginxQps.value = value != null ? Number(value) : null
}

async function fetchAll() {
  loading.value = true
  error.value = ''
  try {
    await Promise.all([fetchCpuUsage(), fetchMemUsage(), fetchNginxQps()])
  } catch (e) {
    console.error(e)
    error.value = '获取监控数据失败，请检查 Prometheus 代理配置。'
  } finally {
    loading.value = false
  }
}

function setupTimer() {
  if (timerId) clearInterval(timerId)
  timerId = setInterval(fetchAll, intervalSec.value * 1000)
}

onMounted(() => {
  fetchAll()
  setupTimer()
})

onBeforeUnmount(() => {
  if (timerId) clearInterval(timerId)
})

watch(intervalSec, () => {
  setupTimer()
})
</script>

<style scoped>
.metrics {
  max-width: 900px;
  margin: 0 auto;
  color: #e5e7eb;
}

.metrics-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1.25rem;
}

.metrics-header h2 {
  margin: 0 0 0.3rem;
  font-size: 1.3rem;
  font-weight: 600;
}

.metrics-subtitle {
  margin: 0;
  font-size: 0.9rem;
  color: #9ca3af;
}

.metrics-controls {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.metrics-interval-label {
  font-size: 0.85rem;
  color: #cbd5f5;
}

.metrics-interval-label select {
  margin-left: 0.25rem;
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  border: 1px solid rgba(75, 85, 99, 0.9);
  background: rgba(15, 23, 42, 0.95);
  color: #e5e7eb;
  font-size: 0.8rem;
}

.metrics-refresh {
  padding: 0.3rem 0.9rem;
  border-radius: 999px;
  border: none;
  cursor: pointer;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  color: #022c22;
  font-size: 0.85rem;
  font-weight: 500;
}

.metrics-refresh:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.metrics-grid {
  margin-top: 1.25rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.metric-card {
  padding: 1rem 1rem 1.1rem;
  border-radius: 0.9rem;
  background: radial-gradient(circle at top, #020617 0, #020617 45%, #030712 100%);
  border: 1px solid rgba(55, 65, 81, 0.9);
  box-shadow: 0 20px 40px -28px rgba(15, 23, 42, 1);
}

.metric-card h3 {
  margin: 0 0 0.3rem;
  font-size: 0.95rem;
  font-weight: 500;
}

.metric-value {
  margin: 0.1rem 0 0.3rem;
  font-size: 1.4rem;
  font-weight: 600;
}

.metric-desc {
  margin: 0;
  font-size: 0.8rem;
  color: #9ca3af;
}

.metrics-error {
  margin-top: 0.8rem;
  font-size: 0.85rem;
  color: #fecaca;
}
</style>
