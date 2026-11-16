import { Fragment, useCallback, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Eraser, Trash2, Zap } from 'lucide-react'
import { api } from '../api'
import { useApp } from '../store'
import { Chip, Empty, Panel, Scramble, Skeleton, Toggle } from '../components/ui'
import { fmtNum, fmtS } from '../format'

const COLORS = ['#8b5cf6', '#22d3ee', '#818cf8', '#34d399', '#fbbf24', '#94a3b8', '#f472b6']

/** Animates the leading number of a formatted value ("1.23 s", "21.6", "1,024") up from zero. */
function CountUp({ value }) {
  const str = String(value)
  const m = str.match(/^(\d[\d,]*\.?\d*)(.*)$/)
  const [out, setOut] = useState(m ? `0${m[2]}` : str)
  useEffect(() => {
    if (!m || matchMedia('(prefers-reduced-motion: reduce)').matches) { setOut(str); return }
    const target = parseFloat(m[1].replace(/,/g, ''))
    const dec = (m[1].split('.')[1] || '').length
    const t0 = performance.now()
    let raf
    const step = (t) => {
      const p = Math.min(1, (t - t0) / 1000)
      const e = 1 - Math.pow(1 - p, 4)
      setOut((target * e).toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) + m[2])
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [str])
  return out
}

function Tile({ label, value, sub, tone }) {
  return <div className={`tile ${tone || ''}`}><div className="tile-l">{label}</div><div className="tile-v"><CountUp value={value} /></div><div className="tile-s">{sub}</div></div>
}

function Stack({ segments }) {
  const segs = segments.filter(([, v]) => v && v > 0)
  const total = segs.reduce((a, [, v]) => a + v, 0)
  if (!total) return <p className="hint">No measurements yet.</p>
  return (
    <>
      <div className="stack">{segs.map(([l, v], i) => (
        <motion.i key={l} title={`${l}: ${fmtS(v)}`} style={{ background: COLORS[i % COLORS.length] }}
          initial={{ width: 0 }} animate={{ width: `${(v / total) * 100}%` }} transition={{ duration: .7, delay: i * .06 }} />))}</div>
      <div className="legend">{segs.map(([l, v], i) => <span key={l}><b style={{ background: COLORS[i % COLORS.length] }} />{l}<em>{fmtS(v)}</em></span>)}</div>
      <div className="stack-total">Total {fmtS(total)}</div>
    </>
  )
}

function Performance() {
  const { conv, source, sources } = useApp()
  const [d, setD] = useState(null)
  useEffect(() => {
    let live = true
    api.get(`/insights/${conv ? `?conversation=${conv.id}` : ''}`).then((r) => live && setD(r)).catch(() => live && setD({ error: true }))
    return () => { live = false }
  }, [conv?.id, conv?.messages.length])
  if (!d) return <div className="stackv"><Skeleton h={90} r={14} /><Skeleton h={140} r={14} /></div>
  if (d.error) return <Empty title="Could not load measurements">Is the backend running?</Empty>
  if (!d.last && !sources.length) return <Empty title="No measurements yet">Index a video and ask a question — real timings show up here.</Empty>
  const l = d.last, tr = d.latest_trace
  const maxTps = Math.max(1, ...d.history.map((h) => h.tps || 0))
  return (
    <div className="stackv">
      {l && (
        <div className="tiles">
          <Tile label="Retrieval" value={fmtS(l.retrieval_s)} sub="query embed + MMR" />
          <Tile label="Model load" value={l.model_load_s ? fmtS(l.model_load_s) : 'cached'} sub={l.model_load_s ? 'this query' : 'already in memory'} />
          <Tile label="First token" value={fmtS(l.ttft_s)} sub="time to first token" tone="acc" />
          <Tile label="Generation" value={fmtS(l.gen_s)} sub="total decode time" />
          <Tile label="Tokens" value={fmtNum(l.tokens)} sub="generated" />
          <Tile label="Speed" value={l.tps ? l.tps.toFixed(1) : '—'} sub="tokens / second" tone="ok" />
        </div>
      )}
      {tr && (
        <Panel title="Where the time went" sub="Latest answer in this conversation, end to end.">
          <Stack segments={[['Query translation', tr.bridge_s], ['Query embedding', tr.qembed_s], ['MMR retrieval', tr.mmr_s], ['Context build', tr.context_s], ['Model load', tr.model_load_s], ['Prefill (to first token)', tr.ttft_s], ['Decode', Math.max((tr.gen_s || 0) - (tr.ttft_s || 0), 0)]]} />
        </Panel>
      )}
      {source && (
        <Panel title={`Indexing · ${source.title || source.video_id}`} sub={`${source.n_chunks.toLocaleString()} chunks · ${source.n_chars.toLocaleString()} characters`}>
          <Stack segments={[['Transcript fetch', source.timings.fetch], ['Splitting', source.timings.split], ['Embedding', source.timings.embed], ['FAISS build + save', source.timings.faiss]]} />
        </Panel>
      )}
      {d.history.length > 0 && (
        <Panel title="Query history" sub={`Most recent ${d.history.length} of ${d.total_queries} (stored in Postgres).`}>
          <div className="tablewrap"><table className="hist"><thead><tr><th>#</th><th>Video</th><th>Retrieval</th><th>TTFT</th><th>Generation</th><th>Tokens</th><th>Tok/s</th></tr></thead>
            <tbody>{d.history.map((h, i) => (
              <tr key={h.id}><td>{d.total_queries - i}</td><td>{h.video_id}</td><td>{fmtS(h.retrieval_s)}</td><td>{fmtS(h.ttft_s)}</td><td>{fmtS(h.gen_s)}</td><td>{fmtNum(h.tokens)}</td>
                <td>{h.tps ? <><i className="bar" style={{ width: `${(h.tps / maxTps) * 70}px` }} />{h.tps.toFixed(1)}</> : '—'}</td></tr>))}</tbody></table></div>
        </Panel>
      )}
    </div>
  )
}

function ProviderCard({ title, m, onTest, busy }) {
  const status = !m.key_set ? <Chip kind="err">no key</Chip> : m.verified ? <Chip kind="ok">verified</Chip> : <Chip kind="acc">key set</Chip>
  return (
    <Panel title={title} right={status}>
      <dl className="kv">
        <dt>Provider</dt><dd>{m.provider}</dd><dt>Model</dt><dd className="mono">{m.model}</dd>
        {m.dim && <><dt>Dimensions</dt><dd>{m.dim} · L2-normalized</dd></>}
        {m.temperature != null && <><dt>Temperature</dt><dd>{m.temperature}</dd></>}
        <dt>API key</dt><dd className="mono">{m.key_set ? m.key_hint : <>not set — add <b>{m.env}</b> to backend/.env</>}</dd>
      </dl>
      <div className="row-actions"><button className="btn" disabled={busy || !m.key_set} onClick={onTest}><Zap size={15} />{busy ? 'Testing…' : 'Test connection'}</button></div>
    </Panel>
  )
}

function Models() {
  const { config, settings, updateSettings, refreshRuntime, toast } = useApp()
  const [m, setM] = useState(null)
  const [busy, setBusy] = useState('')
  const load = useCallback(() => api.get('/models/').then(setM).catch((e) => toast(e.message, 'err')), [toast])
  useEffect(() => { load() }, [load])
  const test = async (kind, path) => {
    setBusy(kind)
    try { const r = await api.post(path); toast(`Connection OK · ${fmtS(r.seconds)}`); await load(); refreshRuntime() }
    catch (e) { toast(e.message, 'err') } finally { setBusy('') }
  }
  if (!m) return <div className="two"><Skeleton h={220} r={22} /><Skeleton h={220} r={22} /></div>
  const r = config.retrieval
  return (
    <div className="stackv">
      <div className="two">
        <ProviderCard title="Embeddings" m={m.embedding} busy={busy === 'e'} onTest={() => test('e', '/models/embedding/check/')} />
        <ProviderCard title="Answer model" m={m.llm} busy={busy === 'l'} onTest={() => test('l', '/models/llm/check/')} />
      </div>
      <Panel title="Retrieval (locked)" sub="Kept exactly as in your original pipeline.">
        <dl className="kv"><dt>Splitter</dt><dd>RecursiveCharacterTextSplitter · {r.chunk_size} / {r.chunk_overlap}</dd><dt>Search</dt><dd>FAISS · MMR · k={r.k} · fetch_k={r.fetch_k} · lambda_mult={r.lambda_mult}</dd><dt>Grounding</dt><dd>only the retrieved context, otherwise “I don’t know.”</dd></dl>
      </Panel>
      <Panel title="Maximum response length" sub="If answers get cut off, raise it. Reasoning models spend part of this budget thinking.">
        <div className="seg">{config.max_token_options.map((n) => (
          <button key={n} className={settings.max_tokens === n ? 'on' : ''} onClick={() => updateSettings({ max_tokens: n })}>{n}{settings.max_tokens === n && <motion.i layoutId="segpill" />}</button>))}</div>
      </Panel>
    </div>
  )
}

function Preferences() {
  const { settings, updateSettings, toast } = useApp()
  const [diag, setDiag] = useState(null)
  const [confirm, setConfirm] = useState(false)
  const { setConv, refreshConvs, newChat } = useApp()
  useEffect(() => { api.get('/diagnostics/').then(setDiag).catch(() => setDiag({ packages: {} })) }, [])
  const wipe = async () => {
    try { await api.del('/conversations/'); setConfirm(false); await newChat(); await refreshConvs(); toast('All chats deleted') } catch (e) { toast(e.message, 'err') }
  }
  return (
    <div className="stackv">
      <Panel>
        <Toggle checked={settings.bridge_boost} onChange={(v) => updateSettings({ bridge_boost: v })}
          label="Boost retrieval for Telugu, Tamil, Kannada, Malayalam, Bengali, Gujarati, Punjabi and Arabic-script questions"
          hint="The chat model translates the question to English for the passage search only (adds about a second). The answer is still written for your original question." />
        <Toggle checked={settings.show_reasoning} onChange={(v) => updateSettings({ show_reasoning: v })}
          label="Show model reasoning (when the model provides it)" hint="Display only. Generation itself is unchanged." />
      </Panel>
      <Panel title="What is and is not supported" sub="Video language and answer language are independent.">
        <dl className="kv wide">
          <dt>Video languages</dt><dd>Only caption tracks YouTube really has for that video (manual or auto-generated). This app does not machine-translate captions.</dd>
          <dt>Answer languages</dt><dd>20 offered. The chat model (Grok or Groq) writes the answer; quality is best in widely used languages and can be weaker in low-resource ones (for example Telugu or Tamil). Check important answers.</dd>
          <dt>Embedding model</dt><dd>Gemini embeddings are multilingual and support cross-language retrieval. Your transcript text and questions are sent to Google and xAI to produce embeddings and answers.</dd>
        </dl>
      </Panel>
      <Panel title="Data" sub="Chats, indexes and metrics are stored in PostgreSQL; vectors in FAISS files on disk — they survive restarts.">
        <div className="row-actions">
          {confirm ? (<><button className="btn danger" onClick={wipe}>Yes, delete all chats</button><button className="btn ghost" onClick={() => setConfirm(false)}>Cancel</button></>)
            : <button className="btn" onClick={() => setConfirm(true)}><Trash2 size={15} />Delete all chats</button>}
          <button className="btn" onClick={() => api.post('/cache/clear/').then(() => toast('Transcript cache cleared')).catch((e) => toast(e.message, 'err'))}><Eraser size={15} />Clear transcript cache</button>
        </div>
      </Panel>
      <Panel title="Diagnostics">
        {!diag ? <Skeleton h={120} r={10} /> : (
          <dl className="kv">{Object.entries(diag.packages).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v || 'not installed'}</dd></Fragment>)}
            {diag.postgres && <><dt>postgresql</dt><dd>{diag.postgres}</dd></>}<dt>Streaming</dt><dd>{diag.streaming ? 'available' : 'unavailable'}</dd></dl>)}
      </Panel>
    </div>
  )
}

export default function InsightsView() {
  const [tab, setTab] = useState('performance')
  const tabs = [['performance', 'Performance'], ['models', 'Models'], ['prefs', 'Preferences']]
  return (
    <div className="page"><div className="page-col">
      <h2 className="view-title"><Scramble text="Insights" /></h2>
      <p className="view-sub">Real measurements. No estimates.</p>
      <div className="seg tabs">{tabs.map(([k, l]) => (
        <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{l}{tab === k && <motion.i layoutId="tabpill" />}</button>))}</div>
      {tab === 'performance' && <Performance />}
      {tab === 'models' && <Models />}
      {tab === 'prefs' && <Preferences />}
    </div></div>
  )
}
