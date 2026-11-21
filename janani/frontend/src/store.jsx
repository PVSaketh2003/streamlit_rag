import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, setUnauthorizedHandler, streamSSE } from './api'
import { emptyStages, QRY_STAGES, splitReasoning } from './format'

const Ctx = createContext(null)
export const useApp = () => useContext(Ctx)

export function AppProvider({ children }) {
  const [boot, setBoot] = useState({ loading: true, error: null, code: null })
  const [config, setConfig] = useState(null)
  const [settings, setSettings] = useState(null)
  const [runtime, setRuntime] = useState(null)
  const [sources, setSources] = useState([])
  const [convs, setConvs] = useState([])
  const [conv, setConv] = useState(null)
  const [view, setView] = useState('chat')
  const [navOpen, setNavOpen] = useState(false)
  const [live, setLive] = useState(null)
  const [toasts, setToasts] = useState([])
  const [account, setAccount] = useState(null)
  const [burst, setBurst] = useState(0)
  const [palette, setPalette] = useState(false)
  const abortRef = useRef(null)
  const convRef = useRef(null)
  convRef.current = conv

  const toast = useCallback((message, kind = 'ok') => {
    const id = Math.random().toString(36).slice(2)
    setToasts((t) => [...t.slice(-3), { id, message, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'err' ? 6500 : 3200)
  }, [])

  const refreshConvs = useCallback(async () => {
    const r = await api.get('/conversations/')
    setConvs(r.results)
  }, [])
  const refreshSources = useCallback(async () => {
    const r = await api.get('/sources/')
    setSources(r.results)
    return r.results
  }, [])
  const refreshRuntime = useCallback(async () => {
    try { setRuntime((await api.get('/health/')).runtime) } catch { /* shown via boot/health */ }
  }, [])

  const openConv = useCallback(async (id) => {
    const c = await api.get(`/conversations/${id}/`)
    setConv(c)
    setView('chat')
    setNavOpen(false)
    return c
  }, [])

  const init = useCallback(async () => {
    setBoot({ loading: true, error: null, code: null })
    try {
      const cfg = await api.get('/config/')
      setConfig(cfg); setSettings(cfg.settings); setRuntime(cfg.runtime); setAccount(cfg.account || null)
      const [, list] = await Promise.all([refreshSources(), api.get('/conversations/')])
      setConvs(list.results)
      const c = list.results[0]
        ? await api.get(`/conversations/${list.results[0].id}/`)
        : await api.post('/conversations/', {})
      setConv(c)
      setBoot({ loading: false, error: null, code: null })
    } catch (e) {
      setBoot({ loading: false, error: e.message, code: e.code || null })
    }
  }, [refreshSources])

  useEffect(() => { init() }, [init])
  // A request found the session gone (expired / signed out elsewhere): show the sign-in page.
  useEffect(() => {
    setUnauthorizedHandler(() => { setAccount(null); setLive(null); setBoot({ loading: false, error: null, code: 'auth_required' }) })
    return () => setUnauthorizedHandler(null)
  }, [])
  useEffect(() => {
    const t = setInterval(refreshRuntime, 8000)
    return () => clearInterval(t)
  }, [refreshRuntime])

  const source = useMemo(() => sources.find((s) => s.id === conv?.source_id) || null, [sources, conv])

  const newChat = useCallback(async (sourceId) => {
    const cur = convRef.current
    const sid = sourceId ?? cur?.source_id ?? null
    const c = await api.post('/conversations/', sid != null ? { source_id: sid } : {})
    setConv(c); setView('chat'); setNavOpen(false)
    refreshConvs()
    return c
  }, [refreshConvs])

  const renameConv = useCallback(async (id, title) => {
    await api.patch(`/conversations/${id}/`, { title })
    setConvs((l) => l.map((c) => (c.id === id ? { ...c, title } : c)))
    setConv((c) => (c?.id === id ? { ...c, title } : c))
  }, [])

  const deleteConv = useCallback(async (id) => {
    await api.del(`/conversations/${id}/`)
    const list = (await api.get('/conversations/')).results
    setConvs(list)
    if (convRef.current?.id === id) {
      setConv(list[0] ? await api.get(`/conversations/${list[0].id}/`) : await api.post('/conversations/', {}))
    }
  }, [])

  const clearConv = useCallback(async (id) => {
    const c = await api.post(`/conversations/${id}/clear/`, {})
    setConv(c); refreshConvs()
  }, [refreshConvs])

  /** Bind a source to the active chat; a chat that already has messages never switches video. */
  const useSource = useCallback(async (sourceId) => {
    const cur = convRef.current
    if (cur && cur.messages.length === 0) {
      const c = await api.patch(`/conversations/${cur.id}/`, { source_id: sourceId })
      setConv(c)
    } else {
      await newChat(sourceId)
    }
    setView('chat')
  }, [newChat])

  const removeSource = useCallback(async (id) => {
    await api.del(`/sources/${id}/`)
    await refreshSources()
    setConv((c) => (c?.source_id === id ? { ...c, source_id: null } : c))
    refreshRuntime()
  }, [refreshSources, refreshRuntime])

  const updateSettings = useCallback(async (patch) => {
    setSettings((s) => ({ ...s, ...patch }))
    try { setSettings(await api.patch('/settings/', patch)) }
    catch (e) { toast(e.message, 'err'); setSettings((await api.get('/settings/'))) }
  }, [toast])

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const logout = useCallback(async () => {
    abortRef.current?.abort()
    try { await api.post('/auth/logout/', {}) } catch { /* cookie is cleared server-side when reachable */ }
    setConv(null); setConvs([]); setSources([]); setLive(null); setAccount(null); setView('chat'); setNavOpen(false)
    setBoot({ loading: false, error: null, code: 'auth_required' })
  }, [])

  const ask = useCallback(async (question, regenerate = false) => {
    const cur = convRef.current
    if (!cur || abortRef.current) return
    if (!regenerate && !question.trim()) return
    const ctrl = new AbortController()
    abortRef.current = ctrl
    let raw = ''
    let frame = 0
    const state = { stages: emptyStages(QRY_STAGES), status: 'Embedding your question', t0: Date.now(), question }
    const push = () => { frame = 0; setLive({ ...state, raw, stages: { ...state.stages } }) }
    const schedule = () => { if (!frame) frame = requestAnimationFrame(push) }
    if (regenerate) {
      setConv((c) => {
        const m = [...c.messages]
        while (m.length && m[m.length - 1].role === 'assistant') m.pop()
        return { ...c, messages: m }
      })
    }
    push()
    const names = { qembed: 'Embedding your question', mmr: 'Finding the best passages', context: 'Building context', infer: 'Thinking' }
    try {
      await streamSSE(`/conversations/${cur.id}/ask/`, { question, regenerate }, (name, d) => {
        if (name === 'start') {
          setConv((c) => ({ ...c, title: d.title, messages: regenerate ? c.messages : [...c.messages, d.user_message] }))
        } else if (name === 'stage') {
          state.stages[d.key] = { status: d.status, ms: d.ms, detail: d.detail }
          if (d.status === 'running' && names[d.key]) {
            state.status = !d.detail || d.detail === 'streaming' ? names[d.key] : `${names[d.key]} · ${d.detail}`
          }
          schedule()
        } else if (name === 'token') {
          raw += d.t
          const s = splitReasoning(raw)
          state.status = s.thinking ? 'Reasoning' : 'Writing'
          schedule()
        } else if (name === 'done') {
          setConv((c) => ({ ...c, messages: [...c.messages, d.message] }))
        } else if (name === 'error') {
          setConv((c) => ({ ...c, messages: [...c.messages, d.assistant_message] }))
        }
      }, ctrl.signal)
    } catch (e) {
      if (e.name !== 'AbortError') toast(e.message, 'err')
    } finally {
      if (frame) cancelAnimationFrame(frame)
      abortRef.current = null
      setLive(null)
      try {
        setConv(await api.get(`/conversations/${cur.id}/`))
        refreshConvs(); refreshRuntime()
      } catch { /* offline: keep local state */ }
    }
  }, [toast, refreshConvs, refreshRuntime])

  /** Index a video; onStage(key, patch) receives live pipeline progress. */
  const indexVideo = useCallback(async ({ url, videoLang, text, langCode }, onEvent) => {
    let result = null
    let failure = null
    const [path, body] = text != null
      ? ['/sources/manual/', { url, text, lang_code: langCode }]
      : ['/sources/index/', { url, video_lang: videoLang }]
    await streamSSE(path, body, (name, d) => {
      onEvent?.(name, d)
      if (name === 'done') result = d
      if (name === 'error') failure = d
    })
    if (failure) throw new ApiError(failure.message, 0, failure.code)
    await refreshSources()
    if (result) { await useSource(result.source.id); if (!result.reused) setBurst((b) => b + 1) }
    refreshRuntime()
    return result
  }, [refreshSources, useSource, refreshRuntime])

  const value = {
    boot, init, config, settings, updateSettings, runtime, refreshRuntime,
    sources, refreshSources, source, convs, conv, setConv, openConv, newChat, renameConv, deleteConv, clearConv,
    useSource, removeSource, view, setView, navOpen, setNavOpen, live, ask, stop, indexVideo,
    toasts, toast, refreshConvs, burst, palette, setPalette, account, logout,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
