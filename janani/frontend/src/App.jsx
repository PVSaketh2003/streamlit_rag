import { AnimatePresence, motion } from 'framer-motion'
import { Loader2, ServerCrash } from 'lucide-react'
import { useApp } from './store'
import { useFx } from './fx'
import Stickers from './components/Stickers'
import { Brand, Drawer, TopBar } from './components/TopBar'
import Login from './components/Login'
import Palette, { Burst } from './components/Palette'
import { Toasts, ErrorBoundary } from './components/ui'
import ChatView from './views/ChatView'
import SourcesView from './views/SourcesView'
import InsightsView from './views/InsightsView'

function Splash({ error, onRetry }) {
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
  if (boot.code === 'auth_required') return <Login onDone={init} />
  if (boot.loading || boot.error) return <Splash error={boot.error} onRetry={init} />
  const View = { chat: ChatView, sources: SourcesView, insights: InsightsView }[view] || ChatView
  return (
    <div className={`shell ${live ? 'is-live' : ''}`}>
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <div className="halo" aria-hidden />
      <div className="dots" aria-hidden />
      <div className="grain" aria-hidden />
      <Stickers />
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
