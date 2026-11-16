import { memo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion } from 'framer-motion'
import { Brain, Check, ChevronDown, Copy, Quote, RefreshCw, Sparkles, TriangleAlert } from 'lucide-react'
import { Chip, copyText } from './ui'
import Pipeline from './Pipeline'
import { fmtS, QRY_STAGES } from '../format'

export const Markdown = ({ children }) => (
  <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown></div>
)

function MetaLine({ t, answerName }) {
  return (
    <div className="chips minimeta">
      {t.video_lang && <Chip kind="ok">{t.video_lang} → {answerName}</Chip>}
      {t.q_script && t.v_script && t.q_script !== t.v_script && <Chip kind="acc">{t.q_script} question · cross-language</Chip>}
      {t.bridge_query && <Chip kind="warn">retrieval boost +{fmtS(t.bridge_s)}</Chip>}
      {t.total_s != null && <Chip kind="acc">{fmtS(t.total_s)}</Chip>}
      {t.tps ? <Chip>{t.tps.toFixed(1)} tok/s</Chip> : null}
      {t.tokens != null && <Chip>{t.tokens} tokens</Chip>}
      {t.ttft_s != null && <Chip>first token {fmtS(t.ttft_s)}</Chip>}
    </div>
  )
}

function TranscriptMap({ n, hits }) {
  const cells = Math.min(n, 120)
  const marks = {}
  hits.forEach((h) => { if (h.chunk > 0) { const c = Math.min(cells - 1, Math.floor(((h.chunk - 1) * cells) / n)); (marks[c] ||= []).push(h.rank) } })
  return (
    <>
      <div className="tmap">{Array.from({ length: cells }, (_, i) => marks[i]
        ? <i key={i} className="hit" title={`retrieved #${Math.min(...marks[i])}`}><u>{Math.min(...marks[i])}</u></i> : <i key={i} />)}</div>
      <div className="tmap-axis"><span>start</span><span>transcript position · {n} chunks</span><span>end</span></div>
    </>
  )
}

function Evidence({ t, msgId }) {
  if (!t.sources?.length) return null
  const open = () => {
    const el = document.getElementById(`how-${msgId}`)
    if (!el) return
    el.open = true
    setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
  }
  return (
    <div className="evidence">
      <span className="ev-l"><Quote size={13} />Grounded in</span>
      {t.sources.map((s) => (
        <button key={s.rank} className="ev" onClick={open} title={s.text.slice(0, 220) + (s.text.length > 220 ? '…' : '')}>
          <b>#{s.chunk > 0 ? s.chunk : s.rank}</b>
          {s.cos != null && <i style={{ '--p': `${Math.max(0, Math.min(1, s.cos)) * 100}%` }} />}
          {s.cos != null && <small>{Math.round(s.cos * 100)}%</small>}
        </button>
      ))}
    </div>
  )
}

function HowAnswered({ t, config, msgId }) {
  const answer = config.langs.find((l) => l.code === t.answer_lang)
  const [open, setOpen] = useState(false)
  const stages = Object.fromEntries(QRY_STAGES.map((s) => [s.key, t.stages?.[s.key] || { status: 'idle' }]))
  return (
    <details className="how" id={`how-${msgId}`} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary><ChevronDown size={15} className="chev" />How this was answered<span>{t.sources.length} sources · {fmtS(t.total_s)}</span></summary>
      {open && (
        <div className="how-body">
          <Pipeline stages={QRY_STAGES} state={stages} compact />
          <dl className="kv">
            <dt>Video language</dt><dd>{t.video_lang} · {t.video_lang_kind} captions · {t.video_lang_reason}</dd>
            <dt>Answer language</dt><dd>{answer?.name} ({answer?.native})</dd>
            <dt>Question script</dt><dd>{t.q_script} (transcript script: {t.v_script})</dd>
            <dt>Retrieval query</dt><dd>{t.bridge_query ? <>{t.bridge_query} <em>(translated by the chat model)</em></> : 'the original question'}</dd>
          </dl>
          <div className="sec">Where the evidence sits</div>
          <TranscriptMap n={t.n_chunks} hits={t.sources} />
          <div className="sec">Retrieved chunks (original transcript language)</div>
          {!t.scores_available && <p className="hint">Scores unavailable for this retrieval path.</p>}
          <div className="chunks">{t.sources.map((s) => (
            <article key={s.rank} className="chunk">
              <header><span className="rank">{s.rank}</span><b>{s.chunk > 0 ? `chunk #${s.chunk}` : 'chunk'}</b><small>retrieval order {s.rank}</small></header>
              {s.cos != null && (<div className="score"><span>cosine similarity</span><b>{s.cos.toFixed(3)}</b>
                <div className="score-bar"><motion.i initial={{ width: 0 }} animate={{ width: `${Math.max(0, Math.min(1, s.cos)) * 100}%` }} /></div></div>)}
              <p>{s.text}</p>
            </article>))}</div>
          <details className="pp"><summary>Prompt sent to the model</summary><pre>{t.prompt}</pre></details>
        </div>
      )}
    </details>
  )
}

function Actions({ msg, isLast, onRetry, busy }) {
  const [done, setDone] = useState(false)
  return (
    <div className="actions">
      <button className="btn-mini" onClick={async () => { if (await copyText(msg.content)) { setDone(true); setTimeout(() => setDone(false), 1400) } }}>
        {done ? <Check size={13} /> : <Copy size={13} />}{done ? 'Copied' : 'Copy'}
      </button>
      {isLast && <button className="btn-mini" disabled={busy} onClick={onRetry}><RefreshCw size={13} />Retry</button>}
    </div>
  )
}

export const Avatar = () => <div className="avatar"><Sparkles size={16} /></div>

export const MessageView = memo(function MessageView({ msg, isLast, onRetry, busy, config, showReasoning }) {
  if (msg.role === 'user') {
    return (
      <motion.div className="msg user" initial={{ opacity: 0, y: 46, scale: .9 }} animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: 'spring', stiffness: 380, damping: 26 }} style={{ transformOrigin: 'bottom right' }}>
        <div className={`bubble ${Date.now() - new Date(msg.created).getTime() < 8000 ? 'fresh' : ''}`}>{msg.content}</div>
      </motion.div>
    )
  }
  const t = msg.trace
  const answerName = t && config.langs.find((l) => l.code === t.answer_lang)?.name
  const fresh = Date.now() - new Date(msg.created).getTime() < 8000
  return (
    <motion.div className={`msg ai ${fresh ? 'fresh' : ''}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <Avatar />
      <div className="ai-body">
        {msg.error ? (
          <div className="errbox"><TriangleAlert size={18} /><div><b>Could not generate an answer.</b><br />{msg.error}</div></div>
        ) : (
          <>
            {msg.reasoning && showReasoning && (
              <details className="reasoning"><summary><Brain size={14} />Reasoning</summary><div>{msg.reasoning}</div></details>
            )}
            {msg.content && <Markdown>{msg.content}</Markdown>}
            {t?.finish_reason === 'length' && <div className="warnbox">The response hit the maximum length. Raise it in Insights → Models to allow longer answers.</div>}
            {t?.stopped && <div className="warnbox">Generation was stopped.</div>}
            {t && !t.stopped && <Evidence t={t} msgId={msg.id} />}
            {t && !t.stopped && <MetaLine t={t} answerName={answerName} />}
          </>
        )}
        {(msg.content || msg.error) && <Actions msg={msg} isLast={isLast} onRetry={onRetry} busy={busy} />}
        {t && !t.stopped && <HowAnswered t={t} config={config} msgId={msg.id} />}
      </div>
    </motion.div>
  )
})
