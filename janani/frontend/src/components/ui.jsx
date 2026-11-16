import { Component, useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, Check, ChevronDown, X } from 'lucide-react'
import { useApp } from '../store'

const GLYPHS = 'అఆకగआकखమలகசಕಗമനবকગકਸਕاب字文かなア한글ДЖΩλ#%&*'

/** Text that "decrypts" out of glyphs from many scripts, settling left to right. */
export function Scramble({ text, duration = 900 }) {
  const [out, setOut] = useState(text)
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { setOut(text); return }
    const t0 = performance.now()
    let raf
    const step = (t) => {
      const p = Math.min(1, (t - t0) / duration)
      const settled = Math.floor(p * text.length)
      setOut(text.split('').map((ch, i) => (i < settled || ch === ' ' ? ch : GLYPHS[(Math.random() * GLYPHS.length) | 0])).join(''))
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [text, duration])
  return <span className="scramble" aria-label={text}>{out}</span>
}

export const Chip = ({ kind = '', children, title }) => <span className={`chip ${kind}`} title={title}>{children}</span>

export function Toggle({ checked, onChange, label, hint }) {
  return (
    <label className="toggle-row">
      <span className="toggle-text"><b>{label}</b>{hint && <small>{hint}</small>}</span>
      <button type="button" role="switch" aria-checked={checked} className={`switch ${checked ? 'on' : ''}`}
        onClick={() => onChange(!checked)}><i /></button>
    </label>
  )
}

export function Select({ value, onChange, options, label, compact }) {
  return (
    <label className={`select ${compact ? 'compact' : ''}`}>
      {label && <span className="select-l">{label}</span>}
      <span className="select-w">
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown size={15} />
      </span>
    </label>
  )
}

export function Panel({ title, sub, right, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || right) && (
        <header className="panel-h">
          <div><h4>{title}</h4>{sub && <p className="sub">{sub}</p>}</div>{right}
        </header>
      )}
      {children}
    </section>
  )
}

export const Empty = ({ icon, title, children }) => (
  <div className="empty">
    {icon && <div className="empty-i">{icon}</div>}
    <b>{title}</b>
    {children && <p>{children}</p>}
  </div>
)

export const Skeleton = ({ h = 16, w = '100%', r = 8 }) => (
  <div className="skeleton" style={{ height: h, width: w, borderRadius: r }} />
)

export function Toasts() {
  const { toasts } = useApp()
  return (
    <div className="toasts" aria-live="polite">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div key={t.id} className={`toast ${t.kind}`} layout
            initial={{ opacity: 0, y: 16, scale: .96 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: .96 }}>
            {t.kind === 'err' ? <AlertTriangle size={16} /> : <Check size={16} />}<span>{t.message}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

export function ErrorNote({ children, onClose }) {
  return (
    <div className="errbox" role="alert">
      <AlertTriangle size={18} /><div>{children}</div>
      {onClose && <button className="icon-btn" onClick={onClose} aria-label="Dismiss"><X size={15} /></button>}
    </div>
  )
}

export class ErrorBoundary extends Component {
  state = { error: null }
  static getDerivedStateFromError(error) { return { error } }
  componentDidCatch(e, info) { console.error('UI error', e, info) }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="fatal">
        <div className="fatal-card">
          <AlertTriangle size={28} />
          <h2>Something broke in the interface</h2>
          <p>{String(this.state.error.message || this.state.error)}</p>
          <button className="btn primary" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      </div>
    )
  }
}

export const copyText = async (text) => {
  try { await navigator.clipboard.writeText(text); return true } catch {
    const ta = document.createElement('textarea')
    ta.value = text; document.body.appendChild(ta); ta.select()
    try { return document.execCommand('copy') } finally { ta.remove() }
  }
}
