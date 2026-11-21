const BASE = import.meta.env.VITE_API_URL || '/api'
export class ApiError extends Error {
  constructor(message, status, code, extra) {
    super(message)
    this.status = status
    this.code = code
    this.extra = extra || {}
  }
}

let onUnauthorized = null
/** The store registers a callback so ANY request that finds the session gone sends the user back to sign-in. */
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn }

const NETWORK_MSG = "Can't reach the backend. Make sure the Django server is running on port 8000."

async function failure(res, path = '') {
  let data = null
  try { data = await res.json() } catch { /* not json */ }
  const e = data?.error || {}
  const err = new ApiError(e.message || `Request failed (${res.status})`, res.status, e.code, e)
  if (res.status === 401 && e.code === 'auth_required' && !path.startsWith('/auth/')) onUnauthorized?.()
  return err
}

async function request(method, path, body, signal) {
  let res
  try {
    res = await fetch(BASE + path, {
      method, signal,
      credentials: 'same-origin',
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(NETWORK_MSG, 0, 'network')
  }
  if (res.status === 204) return null
  if (!res.ok) throw await failure(res, path)
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
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
    })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(NETWORK_MSG, 0, 'network')
  }
  if (!res.ok) throw await failure(res, path)
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
