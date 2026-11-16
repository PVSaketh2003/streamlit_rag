import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { KeyRound, Loader2, ServerCrash } from 'lucide-react'
import { useApp } from './store'
import { accessToken } from './api'
import { useFx } from './fx'
import { Brand, Drawer, TopBar } from './components/TopBar'
import Palette, { Burst } from './components/Palette'
import { Toasts, ErrorBoundary } from './components/ui'
import ChatView from './views/ChatView'
import SourcesView from './views/SourcesView'
import InsightsView from './views/InsightsView'

function AccessGate({ onUnlock, wrong }) {
  const [code, setCode] = useState('')
  return (
    <motion.form className="gate" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
      onSubmit={(e) => { e.preventDefault(); if (code.trim()) { accessToken.set(code.trim()); onUnlock() } }}>
      <div className="gate-i"><KeyRound size={22} /></div>
      <h2>Private workspace</h2>
      <p>Enter the access code to open Janani.</p>
      <input type="password" autoFocus value={code} onChange={(e) => setCode(e.target.value)} placeholder="Access code" aria-label="Access code" />
      {wrong && <small className="gate-err">That code didn’t work. Try again.</small>}
      <button className="btn primary wide" disabled={!code.trim()}>Unlock</button>
    </motion.form>
  )
}

function Splash({ error, code, onRetry }) {
  if (code === 'access_required') {
    const wrong = !!accessToken.get()
    return (
      <div className="splash">
        <div className="aurora" aria-hidden><i /><i /><i /></div>
        <Brand />
        <AccessGate wrong={wrong} onUnlock={onRetry} />
      </div>
    )
  }
  return (
    <div className="splash">
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <Brand />
      {error ? (
        <div className="splash-err"><ServerCrash size={26} /><p>{error}</p>
          <code>cd janani/backend &amp;&amp; python manage.py runserver</code>
          <button className="btn primary" onClick={onRetry}>Retry</button></div>
      ) : <p className="hint"><Loader2 size={15} className="spin" /> Starting up…</p>}
    </div>
  )
}

export default function App() {
  const { boot, init, view, conv, live } = useApp()
  useFx()
  if (boot.loading || boot.error) return <Splash error={boot.error} code={boot.code} onRetry={init} />
  const View = { chat: ChatView, sources: SourcesView, insights: InsightsView }[view] || ChatView
  return (
    <div className={`shell ${live ? 'is-live' : ''}`}>
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <div className="halo" aria-hidden />
      <div className="dots" aria-hidden />
      <div className="grain" aria-hidden />
      <div className="cursor-ring" aria-hidden />
      <TopBar />
      <Drawer />
      <main className="main">
        <ErrorBoundary key={view}>
          <AnimatePresence mode="wait">
            <motion.div key={view} className="view" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: .25 }}>
              <View key={view === 'chat' ? conv?.id : view} />
            </motion.div>
          </AnimatePresence>
        </ErrorBoundary>
      </main>
      <Palette />
      <Burst />
      <Toasts />
    </div>
  )
}
