export const fmtS = (s) => (s == null ? '—' : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(2)} s`)
export const fmtMs = (ms) => (ms == null ? '—' : fmtS(ms / 1000))
export const fmtNum = (n) => (n == null ? '—' : n.toLocaleString())

export function splitReasoning(raw) {
  if (raw.includes('</think>')) {
    const [before, ...rest] = raw.split('</think>')
    return { reasoning: before.replace('<think>', '').trim(), answer: rest.join('</think>').trim(), thinking: false }
  }
  if (raw.trimStart().startsWith('<think>')) {
    return { reasoning: raw.trimStart().slice(7).trim(), answer: '', thinking: true }
  }
  return { reasoning: '', answer: raw, thinking: false }
}

export function extractVideoId(value) {
  const t = (value || '').trim()
  if (!t) return null
  if (/^[A-Za-z0-9_-]{11}$/.test(t)) return t
  const m = t.match(/(?:youtube(?:-nocookie)?\.com\/(?:watch\?(?:.*&)?v=|shorts\/|embed\/|live\/|v\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/)
  return m ? m[1] : null
}

export const IDX_STAGES = [
  { key: 'fetch', label: 'Transcript' },
  { key: 'split', label: 'Splitting' },
  { key: 'embed', label: 'Embedding' },
  { key: 'faiss', label: 'FAISS' },
]
export const QRY_STAGES = [
  { key: 'qembed', label: 'Embed query' },
  { key: 'mmr', label: 'MMR search' },
  { key: 'context', label: 'Context' },
  { key: 'infer', label: 'Inference' },
  { key: 'answer', label: 'Answer' },
]
export const emptyStages = (list) =>
  Object.fromEntries(list.map((s) => [s.key, { status: 'idle', ms: null, detail: '' }]))

export function timeAgo(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

export const isEmail = (v) => /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test((v || '').trim())

export function maskEmail(v) {
  const [local = '', domain = ''] = (v || '').trim().split('@')
  return `${local.slice(0, 1)}${'•'.repeat(Math.max(1, Math.min(local.length - 1, 6)))}@${domain}`
}

/** 0 (empty) .. 4 (strong): a hint for the sign-up meter; the server enforces the real rules. */
export function passwordStrength(pw) {
  if (!pw) return 0
  let score = 0
  if (pw.length >= 8) score++
  if (pw.length >= 12) score++
  if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) score++
  if (/\d/.test(pw) && /[^A-Za-z0-9]/.test(pw)) score++
  return Math.min(4, Math.max(pw.length < 8 ? 1 : score, 1))
}

export const fmtClock = (secs) => `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`

/** A friendly first name from an email address: "saketh.pv@x.com" -> "Saketh", "pvsaketh1@x.com" -> "Pvsaketh". */
export function displayName(email) {
  const local = ((email || '').split('@')[0] || '').split(/[._+-]/)[0].replace(/[^A-Za-z\u00C0-\u024F\u0900-\u0DFF]/g, '')
  return local ? local[0].toUpperCase() + local.slice(1).toLowerCase() : 'there'
}
