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
