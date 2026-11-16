import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { MessagesSquare, Trash2 } from 'lucide-react'
import { useApp } from '../store'
import AddVideo from '../components/AddVideo'
import { Chip, Empty, Scramble } from '../components/ui'
import { fmtS } from '../format'
import { Library } from 'lucide-react'

function SourceCard({ s, active }) {
  const { useSource, removeSource, toast } = useApp()
  const [confirm, setConfirm] = useState(false)
  const t = s.timings
  return (
    <motion.article layout className={`panel src ${active ? 'active' : ''}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: .96 }}>
      <div className="src-top">
        <div className="holo"><img src={s.thumbnail} alt="" onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} /></div>
        <div>
          <h4>{s.title || s.video_id} {active && <Chip kind="ok">in use</Chip>}</h4>
          <code>{s.video_id} · {s.lang_code}</code>
          <div className="chips">
            <Chip kind="ok">{s.lang_name} · {s.lang_kind} captions</Chip>
            <Chip kind="acc">{s.n_chunks.toLocaleString()} chunks</Chip>
            <Chip>{s.n_chars.toLocaleString()} chars</Chip>
            <Chip>indexed in {fmtS(s.total_s)}</Chip>
          </div>
          <div className="chips">
            <Chip>fetch {fmtS(t.fetch)}</Chip><Chip>split {fmtS(t.split)}</Chip><Chip>embed {fmtS(t.embed)}</Chip><Chip>faiss {fmtS(t.faiss)}</Chip>
          </div>
        </div>
      </div>
      <details className="pp"><summary>Caption languages on this video ({s.tracks.length})</summary>
        <div className="chips pad">{s.tracks.map((x) => <Chip key={x.code + x.generated} kind={x.code === s.lang_code ? 'acc' : ''}>{x.name} · {x.code}{x.generated ? ' · auto' : ''}</Chip>)}</div>
      </details>
      <div className="row-actions">
        <button className={`btn ${active ? '' : 'primary'}`} onClick={() => useSource(s.id).catch((e) => toast(e.message, 'err'))}><MessagesSquare size={15} />{active ? 'Open chat' : 'Chat with this'}</button>
        {confirm ? (
          <>
            <button className="btn danger" onClick={() => removeSource(s.id).catch((e) => toast(e.message, 'err'))}>Confirm remove</button>
            <button className="btn ghost" onClick={() => setConfirm(false)}>Cancel</button>
          </>
        ) : <button className="btn ghost" onClick={() => setConfirm(true)} title="Delete the index and its stored chunks"><Trash2 size={15} />Remove</button>}
      </div>
    </motion.article>
  )
}

export default function SourcesView() {
  const { sources, conv } = useApp()
  return (
    <div className="page"><div className="page-col">
      <h2 className="view-title"><Scramble text="Sources" /></h2>
      <p className="view-sub">One FAISS index per video and caption language, stored in Postgres + on disk and reused every time. Each conversation uses exactly one source.</p>
      <AddVideo title="Add a video" />
      <div className="grid-gap">
        {sources.length === 0 && <Empty icon={<Library size={22} />} title="Nothing indexed yet">Paste a link above to build your first knowledge base.</Empty>}
        <AnimatePresence>{sources.map((s) => <SourceCard key={s.id} s={s} active={conv?.source_id === s.id} />)}</AnimatePresence>
      </div>
    </div></div>
  )
}
