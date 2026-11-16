const BASE = import.meta.env.VITE_API_URL || '/api'
const TOKEN_KEY = 'janani-access'

export const accessToken = {
  get() { try { return localStorage.getItem(TOKEN_KEY) || '' } catch { return '' } },
  set(v) { try { localStorage.setItem(TOKEN_KEY, v) } catch { /* private mode */ } },
  clear() { try { localStorage.removeItem(TOKEN_KEY) } catch { /* private mode */ } },
}

function headers(extra = {}) {
  const t = accessToken.get()
  return t ? { ...extra, 'X-Access-Token': t } : extra
}

export class ApiError extends Error {
  constructor(message, status, code, extra) {
    super(message)
    this.status = status
    this.code = code
    this.extra = extra || {}
  }
}

const NETWORK_MSG = "Can't reach the backend. Make sure the Django server is running on port 8000."

async function failure(res) {
  let data = null
  try { data = await res.json() } catch { /* not json */ }
  const e = data?.error || {}
  return new ApiError(e.message || `Request failed (${res.status})`, res.status, e.code, e)
}

async function request(method, path, body, signal) {
  let res
  try {
    res = await fetch(BASE + path, {
      method, signal,
      headers: headers(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(NETWORK_MSG, 0, 'network')
  }
  if (res.status === 204) return null
  if (!res.ok) throw await failure(res)
  return res.json()
}

export const api = {
  get: (p, signal) => request('GET', p, undefined, signal),
  post: (p, b = {}, signal) => request('POST', p, b, signal),
  patch: (p, b, signal) => request('PATCH', p, b, signal),
  del: (p, signal) => request('DELETE', p, undefined, signal),
}

/** POST and consume a Server-Sent Events stream. Calls onEvent(name, data) for each event. */
export async function streamSSE(path, body, onEvent, signal) {
  let res
  try {
    res = await fetch(BASE + path, {
      method: 'POST', signal,
      headers: headers({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
      body: JSON.stringify(body),
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(NETWORK_MSG, 0, 'network')
  }
  if (!res.ok) throw await failure(res)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  const flush = (block) => {
    let name = 'message'
    const data = []
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim()
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
    }
    if (!data.length) return
    try { onEvent(name, JSON.parse(data.join('\n'))) } catch (e) { console.warn('bad SSE frame', e) }
  }
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n\n')) !== -1) {
      flush(buf.slice(0, i))
      buf = buf.slice(i + 2)
    }
  }
  if (buf.trim()) flush(buf)
}
