import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { BarChart3, Brain, CornerDownLeft, Library, MessageSquare, Plus, Search, Video } from 'lucide-react'
import { useApp } from '../store'

/** ⌘K command palette: jump anywhere, start a chat, switch video. */
export default function Palette() {
  const { palette, setPalette, setView, newChat, sources, useSource, settings, updateSettings, toast } = useApp()
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const input = useRef(null)

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette((p) => !p) }
      if (e.key === 'Escape') setPalette(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setPalette])
  useEffect(() => { if (palette) { setQ(''); setI(0); setTimeout(() => input.current?.focus(), 30) } }, [palette])

  const items = useMemo(() => {
    const base = [
      { id: 'new', label: 'New conversation', icon: Plus, run: () => newChat() },
      { id: 'chat', label: 'Go to Chat', icon: MessageSquare, run: () => setView('chat') },
      { id: 'sources', label: 'Go to Sources', icon: Library, run: () => setView('sources') },
      { id: 'insights', label: 'Go to Insights', icon: BarChart3, run: () => setView('insights') },
      { id: 'reason', label: `${settings?.show_reasoning ? 'Hide' : 'Show'} model reasoning`, icon: Brain, run: () => updateSettings({ show_reasoning: !settings.show_reasoning }) },
      ...sources.map((s) => ({ id: `s${s.id}`, label: `Chat with: ${s.title || s.video_id}`, hint: s.lang_name, icon: Video, run: () => useSource(s.id) })),
    ]
    const t = q.trim().toLowerCase()
    return t ? base.filter((x) => x.label.toLowerCase().includes(t)) : base
  }, [q, sources, settings, newChat, setView, useSource, updateSettings])

  const run = (it) => { setPalette(false); Promise.resolve(it?.run()).catch((e) => toast(e.message, 'err')) }
  return (
    <AnimatePresence>
      {palette && (
        <motion.div className="pal-wrap" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={() => setPalette(false)}>
          <motion.div className="pal" role="dialog" aria-label="Command palette" initial={{ y: -16, scale: .97 }} animate={{ y: 0, scale: 1 }} exit={{ y: -10, scale: .98 }}
            onMouseDown={(e) => e.stopPropagation()}>
            <div className="pal-in"><Search size={17} />
              <input ref={input} value={q} placeholder="Type a command or search sources…" onChange={(e) => { setQ(e.target.value); setI(0) }}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') { e.preventDefault(); setI((x) => Math.min(x + 1, items.length - 1)) }
                  if (e.key === 'ArrowUp') { e.preventDefault(); setI((x) => Math.max(x - 1, 0)) }
                  if (e.key === 'Enter') run(items[i])
                }} /></div>
            <ul>{items.map((it, k) => (
              <li key={it.id}><button className={k === i ? 'on' : ''} onMouseEnter={() => setI(k)} onClick={() => run(it)}>
                <it.icon size={16} /><span>{it.label}</span>{it.hint && <small>{it.hint}</small>}{k === i && <CornerDownLeft size={14} />}</button></li>))}
              {items.length === 0 && <li className="pal-none">No matches</li>}
            </ul>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

/** Particle burst celebrating a finished index. */
export function Burst() {
  const { burst } = useApp()
  const [parts, setParts] = useState([])
  useEffect(() => {
    if (!burst) return
    const colors = ['#6366f1', '#c04cf0', '#ff7a59', '#22c1a0', '#f5b301']
    setParts(Array.from({ length: 46 }, (_, k) => {
      const a = (k / 46) * Math.PI * 2 + Math.random() * .4, d = 120 + Math.random() * 260
      return { k, x: Math.cos(a) * d, y: Math.sin(a) * d - 60, c: colors[k % colors.length], s: 5 + Math.random() * 8, r: Math.random() * 360 }
    }))
    const t = setTimeout(() => setParts([]), 1600)
    return () => clearTimeout(t)
  }, [burst])
  return (
    <div className="burst" aria-hidden>
      {parts.map((p) => (
        <motion.i key={`${burst}-${p.k}`} style={{ background: p.c, width: p.s, height: p.s * (p.k % 3 ? 1 : 2.2) }}
          initial={{ x: 0, y: 0, opacity: 1, rotate: 0, scale: 1 }} animate={{ x: p.x, y: p.y + 140, opacity: 0, rotate: p.r, scale: .5 }}
          transition={{ duration: 1.3, ease: [.15, .7, .3, 1] }} />
      ))}
    </div>
  )
}
