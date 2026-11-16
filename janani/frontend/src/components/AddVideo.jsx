import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, ClipboardPaste, ExternalLink, Languages, Link2, Loader2, Play, ShieldAlert, Sparkles } from 'lucide-react'
import { api, ApiError } from '../api'
import { useApp } from '../store'
import { emptyStages, extractVideoId, IDX_STAGES } from '../format'
import Pipeline from './Pipeline'
import { Chip, ErrorNote, Select } from './ui'

export function langOptions(config, withAuto) {
  const base = config.langs.map((l) => ({ value: l.code, label: l.native === l.name ? l.name : `${l.native} · ${l.name}` }))
  return withAuto
    ? [{ value: 'auto', label: 'Auto-detect (recommended)' }, ...base, { value: 'other', label: 'Other / any available' }]
    : base
}

export function LanguageBridge({ videoLang, onVideoLang, answerLang, onAnswerLang, config, locked }) {
  const a = config.langs.find((l) => l.code === answerLang)
  const v = config.langs.find((l) => l.code === videoLang)
  return (
    <div className="bridge">
      <div className="bridge-nodes">
        <div className="bnode">
          <span className="bk">Video language</span>
          <span className="bg">{videoLang === 'auto' ? 'Auto' : videoLang === 'other' ? 'Any' : v?.native}</span>
          {!locked && <Select compact value={videoLang} onChange={onVideoLang} options={langOptions(config, true)} />}
        </div>
        <div className="bridge-arrow"><i /><Languages size={16} /></div>
        <div className="bnode ans">
          <span className="bk">Answer language</span>
          <span className="bg">{a?.native}</span>
          <Select compact value={answerLang} onChange={onAnswerLang} options={langOptions(config, false)} />
        </div>
      </div>
    </div>
  )
}

function PastePanel({ videoId, config, defaultLang, busy, onSubmit, reason }) {
  const [text, setText] = useState('')
  const [lang, setLang] = useState(defaultLang)
  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  return (
    <motion.div className="paste" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
      {reason && <div className="paste-why"><ShieldAlert size={16} /><span>{reason}</span></div>}
      <ol className="paste-steps">
        <li>Open the video on YouTube <a href={`https://www.youtube.com/watch?v=${videoId}`} target="_blank" rel="noreferrer">open <ExternalLink size={12} /></a></li>
        <li>Below the video: <b>…more</b> → <b>Show transcript</b></li>
        <li>Select all the transcript text, copy it, and paste it here. Timestamps are removed automatically.</li>
      </ol>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={7} disabled={busy}
        placeholder="Paste the transcript here…" aria-label="Transcript text" />
      <div className="paste-bar">
        <Select compact value={lang} onChange={setLang} options={langOptions(config, false).map((o) => ({ ...o, label: `Written in ${o.label.split(' · ').pop()}` }))} />
        <span className="muted small">{words.toLocaleString()} words</span>
        <span className="grow" />
        <button className="btn primary" disabled={busy || words < 15} onClick={() => onSubmit(text, lang)}>
          {busy ? <Loader2 size={16} className="spin" /> : <ClipboardPaste size={16} />}Index pasted transcript
        </button>
      </div>
    </motion.div>
  )
}

export default function AddVideo({ title = 'Paste a YouTube link', subtitle, onDone }) {
  const { config, settings, updateSettings, indexVideo, toast } = useApp()
  const [url, setUrl] = useState('')
  const [preview, setPreview] = useState(null)
  const [peek, setPeek] = useState({ loading: false, error: null })
  const [busy, setBusy] = useState(false)
  const [stages, setStages] = useState(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState(null)
  const [alt, setAlt] = useState(null)
  const [paste, setPaste] = useState(null)   // null | { reason }
  const seq = useRef(0)
  const id = extractVideoId(url)

  // Live preview: debounce, then ask the backend which captions exist.
  useEffect(() => {
    setPreview(null); setError(null); setAlt(null); setPaste(null)
    if (!id) { setPeek({ loading: false, error: url.trim() ? 'Not a recognised YouTube video link yet…' : null }); return }
    const my = ++seq.current
    setPeek({ loading: true, error: null })
    const t = setTimeout(async () => {
      try {
        const r = await api.post('/video/inspect/', { url })
        if (my === seq.current) {
          setPreview(r); setPeek({ loading: false, error: null })
          if (r.blocked && !r.existing.length) setPaste({ reason: 'YouTube is blocking caption downloads from this network right now. You can still index this video by pasting its transcript.' })
        }
      } catch (e) {
        if (my === seq.current) setPeek({ loading: false, error: e.message })
      }
    }, 450)
    return () => clearTimeout(t)
  }, [url, id])

  async function run(videoLang = settings.video_lang, pasted = null) {
    if (busy) return
    setBusy(true); setError(null); setAlt(null)
    setStages(emptyStages(IDX_STAGES)); setNote('')
    try {
      const args = pasted ? { url, text: pasted.text, langCode: pasted.lang } : { url, videoLang }
      const res = await indexVideo(args, (name, d) => {
        if (name === 'meta') setNote(`${d.reused ? 'Reusing the existing' : 'Indexing'} ${d.track.name} (${d.track.generated ? 'auto-generated' : 'manual'} captions) · ${d.reason}`)
        if (name === 'stage') setStages((s) => ({ ...s, [d.key]: { status: d.status, ms: d.ms, detail: d.detail } }))
      })
      toast(res.reused ? 'Index already exists — reusing it.' : `Ready — ${res.source.n_chunks} chunks in ${res.source.total_s.toFixed(1)} s`)
      setUrl(''); setStages(null); setPaste(null)
      onDone?.(res.source)
    } catch (e) {
      setStages(null)
      if (e instanceof ApiError && e.code === 'youtube_blocked') { setPaste({ reason: e.message }); return }
      if (e instanceof ApiError && e.code === 'language_unavailable') setAlt(e.extra)
      setError(e.message)
    } finally { setBusy(false) }
  }

  const altTracks = alt ? [...new Map([...alt.tracks].sort((a, b) => a.generated - b.generated).map((t) => [t.code, t])).values()] : []

  return (
    <motion.div className="addvideo" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
      <div className="addvideo-h"><Sparkles size={16} /><b>{title}</b>{subtitle && <span>{subtitle}</span>}</div>
      <form className={`linkbox ${peek.error && url ? 'bad' : ''}`} onSubmit={(e) => { e.preventDefault(); run() }}>
        <Link2 size={18} />
        <input value={url} onChange={(e) => setUrl(e.target.value)} disabled={busy} spellCheck={false} autoFocus
          placeholder="https://www.youtube.com/watch?v=…  ·  youtu.be/…  ·  shorts  ·  or an 11-character ID"
          aria-label="YouTube link" />
        <button className="btn primary" disabled={busy || !id}>
          {busy ? <Loader2 size={16} className="spin" /> : <Play size={16} />}{busy ? 'Indexing' : 'Index video'}
        </button>
      </form>
      {peek.error && url && <p className="hint warn">{peek.error}</p>}
      {peek.loading && <p className="hint"><Loader2 size={13} className="spin" /> Checking captions…</p>}
      {preview && !busy && (
        <motion.div className="preview" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <img src={preview.thumbnail} alt="" onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
          <div>
            <b>{preview.title || preview.video_id}</b>
            <div className="chips">
              {preview.blocked
                ? <Chip kind="warn">captions unreachable right now</Chip>
                : <Chip kind="ok">{preview.tracks.length} caption {preview.tracks.length === 1 ? 'track' : 'tracks'}</Chip>}
              {preview.existing.length > 0 && <Chip kind="acc">already indexed · {preview.existing.map((s) => s.lang_code).join(', ')}</Chip>}
            </div>
          </div>
          <ArrowRight size={18} className="muted" />
        </motion.div>
      )}
      <LanguageBridge config={config} videoLang={settings.video_lang} answerLang={settings.answer_lang}
        onVideoLang={(v) => updateSettings({ video_lang: v })} onAnswerLang={(v) => updateSettings({ answer_lang: v })} />
      {stages && (<><Pipeline stages={IDX_STAGES} state={stages} />{note && <p className="rib-note">{note}.</p>}</>)}
      {id && !busy && !paste && (
        <button type="button" className="linklike" onClick={() => setPaste({ reason: '' })}><ClipboardPaste size={14} />Or paste the transcript yourself</button>
      )}
      {id && paste && (
        <PastePanel videoId={id} config={config} busy={busy} reason={paste.reason}
          defaultLang={['auto', 'other'].includes(settings.video_lang) ? 'en' : settings.video_lang}
          onSubmit={(text, lang) => run(undefined, { text, lang })} />
      )}
      {error && <ErrorNote onClose={() => { setError(null); setAlt(null) }}>{error}
        {alt && (
          <div className="alt">
            <small>Only caption tracks that really exist can be indexed. Pick one:</small>
            <div className="chips">{altTracks.slice(0, 40).map((t) => (
              <button key={t.code} className="chip btn-chip" onClick={() => run(t.code)}>{t.name}{t.generated ? ' · auto' : ''}</button>
            ))}</div>
          </div>)}
      </ErrorNote>}
    </motion.div>
  )
}
