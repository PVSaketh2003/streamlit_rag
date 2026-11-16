import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { BarChart3, Check, Command, History, Library, MessageSquare, Pencil, Plus, Trash2, X } from 'lucide-react'
import { useApp } from '../store'
import { timeAgo } from '../format'

export const Brand = () => (
  <div className="brand">
    <div className="brand-mark"><svg viewBox="0 0 64 64"><path d="M40 16v22a10 10 0 0 1-20 0" fill="none" stroke="#fff" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" /></svg></div>
    <span className="brand-name">Janani<em>studio</em></span>
  </div>
)

const NAV = [['chat', 'Chat', MessageSquare], ['sources', 'Sources', Library], ['insights', 'Insights', BarChart3]]

export function TopBar() {
  const { view, setView, setNavOpen, runtime, setPalette } = useApp()
  const [hover, setHover] = useState(null)
  const rt = runtime || {}
  return (
    <header className="topbar">
      <div className="tb-left">
        <button className="icon-btn" onClick={() => setNavOpen(true)} aria-label="Conversation history"><History size={18} /></button>
        <Brand />
      </div>
      <nav className="dock" aria-label="Primary" onMouseLeave={() => setHover(null)}>
        {NAV.map(([k, label, Icon]) => (
          <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)} onMouseEnter={() => setHover(k)}>
            {hover === k && view !== k && <motion.i layoutId="dockhover" className="dockhover" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
            {view === k && <motion.i layoutId="dockpill" className="dockpill" transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
            <Icon size={15} /><span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="tb-right">
        <button className="kbd" onClick={() => setPalette(true)} aria-label="Open command palette"><Command size={13} />K</button>
        <button className="status" onClick={() => setView('insights')} title="Provider status — open Insights → Models">
          <i className={`dot ${rt.llm_verified ? 'on' : rt.llm_ready ? 'set' : 'off'}`} />{rt.llm_short || 'Grok'}
          <i className={`dot ${rt.embedding_verified ? 'on' : rt.embedding_ready ? 'set' : 'off'}`} />Gemini
        </button>
      </div>
    </header>
  )
}

function ConvItem({ c, active }) {
  const { openConv, renameConv, deleteConv, toast } = useApp()
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(c.title)
  const [confirm, setConfirm] = useState(false)
  const save = async () => {
    setEditing(false)
    if (title.trim() && title.trim() !== c.title) {
      try { await renameConv(c.id, title.trim()) } catch (e) { toast(e.message, 'err') }
    } else setTitle(c.title)
  }
  if (editing) {
    return (
      <div className="conv editing">
        <input autoFocus value={title} maxLength={80} onChange={(e) => setTitle(e.target.value)} onBlur={save}
          onKeyDown={(e) => { if (e.key === 'Enter') save(); if (e.key === 'Escape') { setTitle(c.title); setEditing(false) } }} />
      </div>
    )
  }
  return (
    <div className={`conv ${active ? 'active' : ''}`}>
      <button className="conv-main" onClick={() => openConv(c.id).catch((e) => toast(e.message, 'err'))}>
        <span className="conv-t">{c.title}</span><small>{timeAgo(c.updated)}</small>
      </button>
      <span className="conv-actions">
        {confirm ? (
          <>
            <button aria-label="Confirm delete" className="danger" onClick={() => deleteConv(c.id).catch((e) => toast(e.message, 'err'))}><Check size={14} /></button>
            <button aria-label="Cancel" onClick={() => setConfirm(false)}><X size={14} /></button>
          </>
        ) : (
          <>
            <button aria-label="Rename" onClick={() => setEditing(true)}><Pencil size={13} /></button>
            <button aria-label="Delete" onClick={() => setConfirm(true)}><Trash2 size={13} /></button>
          </>
        )}
      </span>
    </div>
  )
}

export function Drawer() {
  const { convs, conv, view, newChat, navOpen, setNavOpen, toast, live } = useApp()
  return (
    <AnimatePresence>
      {navOpen && (
        <>
          <motion.div className="scrim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setNavOpen(false)} />
          <motion.aside className="drawer" initial={{ x: -340 }} animate={{ x: 0 }} exit={{ x: -340 }} transition={{ type: 'spring', stiffness: 380, damping: 38 }}>
            <div className="drawer-h"><b>Conversations</b><button className="icon-btn" onClick={() => setNavOpen(false)} aria-label="Close"><X size={17} /></button></div>
            <button className="btn primary wide" disabled={!!live} onClick={() => newChat().catch((e) => toast(e.message, 'err'))}><Plus size={16} />New conversation</button>
            <div className="convs">
              {convs.length === 0 && <p className="side-empty">Your chats will appear here.</p>}
              {convs.map((c) => <ConvItem key={c.id} c={c} active={c.id === conv?.id && view === 'chat'} />)}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
