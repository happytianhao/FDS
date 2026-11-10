<!-- Transparent, theme-aware FDS monitor. All measurements arrive through widget data. -->
<script setup>
import { computed, ref, unref, onMounted, onUnmounted } from 'vue'
import { useWidgetData } from '@whalent/widget-sdk'

const incoming = useWidgetData()
const data = computed(() => { const value = unref(incoming); return unref(value?.data) ?? value ?? {} })
const tab = ref('overview')
const now = ref(Date.now())
let timer
onMounted(() => { timer = setInterval(() => { now.value = Date.now() }, 10000) })
onUnmounted(() => clearInterval(timer))
const exp = computed(() => data.value.experiment ?? {})
const usage = computed(() => data.value.usage ?? {})
const quota = computed(() => data.value.quota ?? {})
const activeRuns = computed(() => exp.value.active_runs ?? (exp.value.active ? [exp.value.active] : []))
const rows = computed(() => exp.value.tasks ?? [])
const windows = computed(() => (quota.value.buckets ?? []).flatMap(bucket =>
  (bucket.windows ?? []).map(window => ({ ...window, bucket: bucket.name, plan: bucket.plan }))))
const percent = computed(() => exp.value.planned ? 100 * (exp.value.completed ?? 0) / exp.value.planned : 0)
const age = computed(() => data.value.updated_at ? Math.max(0, Math.floor((now.value - Date.parse(data.value.updated_at)) / 1000)) : null)
const stale = computed(() => exp.value.status !== 'completed' && age.value !== null && age.value > 180)
const status = computed(() => stale.value ? '数据待刷新' : exp.value.status === 'completed' ? '已结束' : exp.value.coordinator_alive === false ? '需检查' : exp.value.status === 'waiting_review' ? '等待分析' : exp.value.hold_reason ? '故障恢复待确认' : exp.value.status === 'running' ? `运行中 ${activeRuns.value.length}/${exp.value.max_concurrency ?? 1}` : '等待数据')
const fmt = value => value == null ? '—' : Number(value).toLocaleString('zh-CN')
const compact = value => value == null ? '—' : value >= 1e6 ? (value / 1e6).toFixed(2) + 'M' : value >= 1e3 ? (value / 1e3).toFixed(1) + 'K' : String(value)
const clamp = value => Math.max(0, Math.min(100, Number(value) || 0))
const stamp = value => !value ? '—' : new Date(typeof value === 'number' ? value * 1000 : value).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
const windowLabel = mins => mins == null ? '额度窗口' : mins % 1440 === 0 ? `${mins / 1440} 天窗口` : mins % 60 === 0 ? `${mins / 60} 小时窗口` : `${mins} 分钟窗口`
const rowLabel = row => row.case_state === 'blocked' ? '未解锁' : row.case_state === 'locked' ? '等待前置成功' : row.state === 'running' ? '运行中' : row.case_state === 'waiting_review' ? '失败·待分析' : row.case_state === 'retry' ? '等待复跑' : row.success ? '成功' : row.case_state === 'stopped' ? '失败' : row.state === 'pending' ? '待运行' : '未成功'

</script>

<template>
  <section class="widget" aria-label="FDS 实验进度与额度">
    <header>
      <strong>FDS <span>实验看板</span></strong>
      <span class="badge" :class="{ warn: stale || exp.coordinator_alive === false }"><i :class="{ live: exp.status === 'running' && !stale }" />{{ status }}</span>
    </header>
    <nav aria-label="显示内容">
      <button v-for="item in [{ id: 'overview', label: '概览' }, { id: 'tasks', label: '场景' }, { id: 'quota', label: '额度与用量' }]" :key="item.id" :aria-pressed="tab === item.id" :class="{ selected: tab === item.id }" @click="tab = item.id">{{ item.label }}</button>
      <small v-if="exp.planned">{{ fmt(exp.completed) }}/{{ fmt(exp.planned) }}</small>
    </nav>
    <main>
      <template v-if="tab === 'overview'">
        <div class="current">
          <div class="row"><span class="muted">当前运行</span><b>{{ activeRuns.length }}/{{ exp.max_concurrency ?? 1 }} 场景</b></div>
          <div v-for="item in activeRuns" :key="item.task" class="row"><b>{{ item.task }}</b><span class="muted">{{ fmt(item.environment_steps) }} 步 · {{ fmt(item.planner_calls) }}/{{ fmt(exp.call_budget) }} calls</span></div>
          <p v-if="!activeRuns.length" class="hint">{{ exp.status === 'completed' ? '全部结束' : '等待下一个场景' }}</p>
          <div class="muted">{{ exp.benchmark_label ?? '—' }} · {{ exp.split ?? '—' }} ep{{ exp.episode ?? '—' }}</div>
        </div>
        <div class="meter" role="progressbar" :aria-valuenow="exp.completed ?? 0" :aria-valuemax="exp.planned ?? 0" aria-valuemin="0" aria-label="已完成场景"><span :style="{ width: clamp(percent) + '%' }" /></div>
        <div class="stats"><div><b class="success">{{ fmt(exp.successes) }}</b><span>成功</span></div><div><b>{{ fmt(exp.task_failures) }}</b><span>失败</span></div><div><b>{{ fmt(exp.blocked) }}</b><span>未解锁</span></div><div><b>{{ fmt(exp.remaining) }}</b><span>未结束</span></div></div>
        <div class="row strip"><span>本轮 token <b>{{ compact(usage.total_tokens) }}</b></span><span>纠正成功 <b>{{ fmt(usage.recovered_decisions) }}</b></span></div>
        <div class="row quota-peek" v-if="windows.length"><span>{{ windowLabel(windows[0].duration_minutes) }}剩余</span><b :class="{ warning: windows[0].remaining_percent < 15 }">{{ windows[0].remaining_percent ?? '—' }}%</b></div>
        <p v-else class="hint">账户额度：{{ quota.status === 'unknown' ? '暂不可读' : '等待采样' }}</p>
        <p v-if="data.optimizer?.status === 'needs_user_decision'" class="hint warning">有特殊改动等待用户确认</p>
        <p class="hint">{{ exp.model ?? '—' }} · {{ exp.effort ?? '—' }} · {{ exp.policy_label ?? '—' }}</p>
      </template>
      <template v-else-if="tab === 'tasks'">
        <p class="hint">全部为训练集；点击场景查看步数与失败原因。</p>
        <details v-for="row in rows" :key="row.task" class="task">
          <summary><span>{{ row.task }}</span><span :class="{ success: row.success, warning: row.state === 'error' }">{{ rowLabel(row) }}</span></summary>
          <div class="details"><div>{{ fmt(row.planner_calls) }} calls · {{ fmt(row.environment_steps) }} 步 · {{ compact(row.total_tokens) }} token</div><div>拒绝 {{ fmt(row.rejections) }} · 纠正成功 {{ fmt(row.recoveries) }}</div><div v-if="row.outcome_detail_label">{{ row.outcome_detail_label }}</div><div v-if="row.official_status">official: {{ row.official_status }} · {{ row.stop_reason ?? '进行中' }}</div><p v-if="row.error">{{ row.error }}</p></div>
        </details>
      </template>
      <template v-else>
        <div class="section-label">账户额度 <small>{{ quota.buckets?.[0]?.plan ?? '' }}</small></div>
        <div v-for="window in windows" :key="window.bucket + window.name" class="quota">
          <div class="row"><span>{{ window.bucket }} · {{ windowLabel(window.duration_minutes) }}</span><b :class="{ warning: window.remaining_percent < 15 }">剩余 {{ window.remaining_percent ?? '—' }}%</b></div>
          <div class="meter"><span :class="{ low: window.remaining_percent < 15 }" :style="{ width: clamp(window.remaining_percent) + '%' }" /></div>
          <small class="muted">重置 {{ stamp(window.resets_at) }}</small>
        </div>
        <p v-if="!windows.length" class="hint">额度未知；没有从 token 用量推算余额。</p>
        <div v-for="bucket in quota.buckets ?? []" :key="bucket.id" class="row muted"><span>额外 credits</span><span>{{ bucket.credits?.unlimited ? '不限额（服务返回）' : bucket.credits?.balance ?? '未知' }}</span></div>
        <p class="hint">{{ quota.note }}<br>采样 {{ stamp(quota.sampled_at) }}<span v-if="quota.refresh_error"> · 最新读取失败，显示上次记录</span></p>
        <div class="section-label">本轮实验用量</div>
        <div class="usage-grid"><span>输入 token</span><b>{{ fmt(usage.prompt_tokens) }}</b><span>输出 token</span><b>{{ fmt(usage.completion_tokens) }}</b><span>合计</span><b>{{ fmt(usage.total_tokens) }}</b><span>planner calls</span><b>{{ fmt(usage.planner_calls) }}</b><span>环境步数</span><b>{{ fmt(usage.environment_steps) }}</b><span>合约拒绝 / 恢复成功</span><b>{{ fmt(usage.rejections) }} / {{ fmt(usage.recovered_decisions) }}</b></div>
        <p class="hint">{{ usage.note }}<span v-if="usage.pending_requests"> · {{ usage.pending_requests }} 次请求尚无用量回执</span><span v-if="usage.missing_usage"> · {{ usage.missing_usage }} 次用量缺失，合计为下界</span></p>
        <p class="formula">{{ usage.cost_formula ?? '单价未设置，暂不折算金额。' }}</p>
        <p class="hint">上轮 token：{{ fmt(usage.previous_run_tokens) }}，未计入本轮。</p>
      </template>
    </main>
    <footer><span :class="{ warning: stale }">{{ age == null ? '等待数据' : age < 60 ? `${age} 秒前更新` : `${Math.floor(age / 60)} 分钟前更新` }}</span><span :title="data.monitor?.next_check_at">AI 检查 {{ data.monitor?.interval_label ?? '待安排' }}</span></footer>
  </section>
</template>

<style scoped>
.widget { height:100%; min-height:210px; box-sizing:border-box; padding:10px 12px 8px; display:flex; flex-direction:column; gap:7px; background:transparent; color:var(--wr-text); font:12px/1.4 system-ui,sans-serif; font-variant-numeric:tabular-nums; }
header,.row,nav,footer { display:flex; align-items:center; justify-content:space-between; gap:8px; }
header strong { font-size:14px; letter-spacing:.1px; } header strong span { font-size:12px; font-weight:500; color:var(--wr-text2); margin-left:4px; }
.badge { font-size:10px; color:var(--wr-text2); display:flex; align-items:center; gap:5px; }.badge i { width:6px; height:6px; border-radius:50%; background:currentColor; }.badge i.live { background:#15a98a; box-shadow:0 0 0 3px #15a98a18; }
nav { justify-content:flex-start; border-bottom:1px solid var(--wr-border); padding-bottom:4px; gap:3px; } nav small { margin-left:auto; color:var(--wr-text2); } button { color:var(--wr-text2); background:transparent; font:inherit; font-size:11px; border:0; border-radius:5px; padding:3px 7px; cursor:pointer; } button.selected { color:var(--wr-text); background:color-mix(in srgb,var(--wr-text) 8%,transparent); font-weight:600; } button:focus-visible,summary:focus-visible { outline:2px solid #15a98a; outline-offset:2px; }
main { flex:1; min-height:0; overflow:auto; scrollbar-width:thin; }.current { display:grid; gap:3px; }.current .row:first-child b { font-size:13px; }.current .muted { font-size:10px; }
.muted,.hint { color:var(--wr-text2); }.meter { height:5px; margin:7px 0; border-radius:5px; overflow:hidden; background:color-mix(in srgb,var(--wr-text) 10%,transparent); }.meter span { display:block; height:100%; background:#15a98a; border-radius:5px; transition:width .4s ease; }.meter span.low { background:#d38c28; }
.stats { display:grid; grid-template-columns:repeat(4,1fr); margin-bottom:6px; }.stats div { display:flex; align-items:baseline; gap:5px; }.stats b { font-size:17px; font-weight:650; }.stats span { font-size:10px; color:var(--wr-text2); }.success { color:#15a98a; }.warning,.warn { color:#ce8a25; }
.strip { border-top:1px solid var(--wr-border); padding-top:5px; font-size:11px; }.quota-peek { margin-top:5px; font-size:11px; }.hint { font-size:10px; line-height:1.5; margin:5px 0 0; }.section-label { font-size:11px; font-weight:650; margin:7px 0; }.section-label:first-child { margin-top:0; }.section-label small { color:var(--wr-text2); margin-left:5px; font-weight:400; }.quota { margin-bottom:8px; }.quota .row { font-size:11px; }.quota small { font-size:10px; }
.task { border-bottom:1px solid var(--wr-border); }.task summary { cursor:pointer; display:flex; justify-content:space-between; gap:8px; padding:6px 0; font-size:11px; list-style:none; }.task summary::-webkit-details-marker { display:none; }.task summary span:last-child { font-size:10px; }.details { padding:0 0 7px; color:var(--wr-text2); font-size:10px; overflow-wrap:anywhere; }.details p { margin:3px 0; }.usage-grid { display:grid; grid-template-columns:1fr auto; gap:4px; font-size:11px; }.usage-grid span { color:var(--wr-text2); }.usage-grid b { font-weight:550; }.formula { font-size:10px; color:var(--wr-text2); overflow-wrap:anywhere; margin:7px 0; }
footer { flex-shrink:0; border-top:1px solid var(--wr-border); padding-top:5px; color:var(--wr-text2); font-size:9px; } @media(prefers-reduced-motion:reduce){.meter span{transition:none;}}
</style>
