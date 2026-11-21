import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowDown, ArrowUp, Eraser, Square, Video } from 'lucide-react'
import { useApp } from '../store'
import AddVideo, { LanguageBridge, langOptions } from '../components/AddVideo'
import Pipeline from '../components/Pipeline'
import { Avatar, MessageView } from '../components/Message'
import { Chip, Select } from '../components/ui'
import { displayName, QRY_STAGES, splitReasoning } from '../format'

const TICKER = ['Hello', 'నమస్కారం', 'नमस्ते', 'வணக்கம்', 'こんにちは', 'Hola', 'مرحبا', '你好', 'Bonjour', 'Guten Tag', '안녕하세요', 'Olá', 'Привет', 'ನಮಸ್ಕಾರ', 'നമസ്കാരം', 'নমস্কার', 'સત શ્રી અકાલ', 'سلام']
const WORDS = ['lecture', 'podcast', 'keynote', 'tutorial', 'interview']

function Letters({ text, delay = 0 }) {
  return text.split('').map((ch, k) => (
    <motion.span key={k} className={ch === ' ' ? 'ltr sp' : 'ltr'} initial={{ y: '0.9em', opacity: 0, filter: 'blur(8px)' }} animate={{ y: 0, opacity: 1, filter: 'blur(0px)' }}
      transition={{ delay: delay + k * .035, duration: .6, ease: [.2, .8, .2, 1] }}><span className="rp">{ch === ' ' ? '\u00A0' : ch}</span></motion.span>
  ))
}

function Hero({ name, tapes = true, compact = false }) {
  const [i, setI] = useState(0)
  useEffect(() => { const t = setInterval(() => setI((x) => (x + 1) % WORDS.length), 2400); return () => clearInterval(t) }, [])
  // Letters lean away from the cursor (springy, settles back when the pointer leaves).
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || matchMedia('(pointer: coarse)').matches) return
    const els = [...document.querySelectorAll('.hero h1 .rp')]
    const cur = els.map(() => ({ x: 0, y: 0 }))
    let mx = -1e4, my = -1e4, raf = 0
    const tick = () => {
      let moving = false
      els.forEach((el, i) => {
        const r = el.getBoundingClientRect()
        const dx = r.left + r.width / 2 - cur[i].x - mx, dy = r.top + r.height / 2 - cur[i].y - my
        const d = Math.hypot(dx, dy) || 1
        const f = d < 150 ? (1 - d / 150) ** 2 * 28 : 0
        const tx = (dx / d) * f, ty = (dy / d) * f
        cur[i].x += (tx - cur[i].x) * .16; cur[i].y += (ty - cur[i].y) * .16
        if (Math.abs(tx - cur[i].x) > .05 || Math.abs(ty - cur[i].y) > .05) moving = true
        el.style.transform = `translate(${cur[i].x.toFixed(2)}px, ${cur[i].y.toFixed(2)}px) rotate(${(cur[i].x * .7).toFixed(2)}deg)`
      })
      raf = moving ? requestAnimationFrame(tick) : 0
    }
    const move = (e) => { mx = e.clientX; my = e.clientY; if (!raf) raf = requestAnimationFrame(tick) }
    const out = () => { mx = my = -1e4; if (!raf) raf = requestAnimationFrame(tick) }
    window.addEventListener('pointermove', move, { passive: true })
    document.addEventListener('pointerleave', out)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('pointermove', move); document.removeEventListener('pointerleave', out) }
  }, [])
  return (
    <div className={`hero ${compact ? 'compact' : ''}`}>
      <motion.span className="pill" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}><b />{name ? `Hi ${name}` : 'Hello'} · ask in any language, answer in the one you choose</motion.span>
      <h1><Letters text="Talk to any" /> <span className="rot"><AnimatePresence mode="wait"><motion.em key={i} initial={{ y: 30, opacity: 0, filter: 'blur(8px)' }} animate={{ y: 0, opacity: 1, filter: 'blur(0px)' }} exit={{ y: -30, opacity: 0, filter: 'blur(8px)' }} transition={{ duration: .45 }}>{WORDS[i]}</motion.em></AnimatePresence></span></h1>
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: .7 }}>Paste a YouTube link. The video and the answer can be in different languages.</motion.p>
      {tapes && <Tapes words={TICKER} />}
    </div>
  )
}

/** Two crossing ribbons of greetings that scroll in opposite directions. Decorative only. */
function Tapes({ words }) {
  const row = (list) => [...list, ...list].map((w, k) => <span key={k}>{w}<i>✦</i></span>)
  const rev = [...words].reverse()
  return (
    <div className="tapes" aria-hidden="true">
      <div className="tape t1"><div className="tape-track">{row(words)}</div></div>
      <div className="tape t2"><div className="tape-track rev">{row(rev)}</div></div>
    </div>
  )
}

/** Live answer: each new word "bleeds" in from blur. Keys are stable, so only new words animate.
 *  Markdown markers are hidden while streaming; the final message renders full markdown. */
function StreamText({ text }) {
  const clean = text.replace(/\*\*|__|`/g, '').replace(/^#{1,6}\s+/gm, '').replace(/^\s*[-*]\s+/gm, '• ')
  const parts = clean.split(/(\s+)/)
  return (
    <div className="stream md">
      {parts.map((p, i) => (/^\s*$/.test(p) ? p : <span key={i} className="w">{p}</span>))}
      <span className="cursor" />
    </div>
  )
}

function LiveAnswer() {
  const { live, settings } = useApp()
  const [, tick] = useState(0)
  useEffect(() => { const t = setInterval(() => tick((x) => x + 1), 200); return () => clearInterval(t) }, [])
  if (!live) return null
  const { reasoning, answer, thinking } = splitReasoning(live.raw)
  const el = ((Date.now() - live.t0) / 1000).toFixed(1)
  return (
    <div className="msg ai live">
      <Avatar />
      <div className="ai-body">
        <Pipeline stages={QRY_STAGES} state={live.stages} compact />
        {!answer && <div className="think"><i /><i /><i /><span>{live.status} · {el}s</span></div>}
        {thinking && settings.show_reasoning && reasoning && <div className="reason-live">{reasoning.slice(-700)}</div>}
        {answer && <StreamText text={answer} />}
      </div>
    </div>
  )
}

function Composer({ disabled, reason }) {
  const { ask, stop, live, config, settings, updateSettings, conv, clearConv, toast } = useApp()
  const [text, setText] = useState('')
  const ref = useRef(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 180) + 'px' }
  }, [text])
  const send = () => {
    const q = text.trim()
    if (!q || disabled || live) return
    setText(''); ask(q)
  }
  return (
    <div className="composer-wrap">
      <div className={`composer ${disabled ? 'off' : ''} ${live ? 'live' : ''}`}>
        <textarea ref={ref} rows={1} value={text} disabled={disabled} maxLength={4000}
          placeholder={disabled ? reason : 'Ask in any language…'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send() } }} />
        <div className="composer-bar">
          <Select compact value={settings.answer_lang} onChange={(v) => updateSettings({ answer_lang: v })}
            options={langOptions(config, false).map((o) => ({ ...o, label: `Answer in ${o.label.split(' · ').pop()}` }))} />
          <span className="grow" />
          {conv?.messages.length > 0 && !live && (
            <button className="icon-btn" title="Clear conversation" aria-label="Clear conversation"
              onClick={() => clearConv(conv.id).catch((e) => toast(e.message, 'err'))}><Eraser size={16} /></button>
          )}
          {live
            ? <button className="send stop" onClick={stop} aria-label="Stop generating"><Square size={14} fill="currentColor" /></button>
            : <button className="send" onClick={send} disabled={disabled || !text.trim()} aria-label="Send"><ArrowUp size={18} /></button>}
        </div>
      </div>
      <p className="composer-note">Answers are grounded only in the video transcript · Enter to send, Shift+Enter for a new line</p>
    </div>
  )
}

export default function ChatView() {
  const { conv, source, live, ask, config, settings, updateSettings, setView, account } = useApp()
  const scroller = useRef(null)
  const stick = useRef(true)
  const [showJump, setShowJump] = useState(false)
  const msgs = conv?.messages || []

  useEffect(() => {
    const el = scroller.current
    if (!el) return
    if (msgs.length === 0 && !live) el.scrollTo({ top: 0 })
    else if (stick.current) el.scrollTo({ top: el.scrollHeight })
  }, [msgs.length, live?.raw, live?.stages, conv?.id])
  useEffect(() => { stick.current = true }, [conv?.id])

  const onScroll = () => {
    const el = scroller.current
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    stick.current = near
    setShowJump(!near)
  }
  const empty = msgs.length === 0 && !live

  return (
    <div className="chat">
      <header className="chat-head">
        {source ? (
          <button className="srcbar" onClick={() => setView('sources')} title="Manage sources">
            <img src={source.thumbnail} alt="" onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
            <span><b>{source.title || source.video_id}</b><small>{source.lang_name} · {source.lang_kind} captions · {source.n_chunks} chunks</small></span>
          </button>
        ) : <span className="srcbar none"><Video size={15} />No video selected</span>}
        <span className="grow" />
        {conv && conv.title !== 'New conversation' && <Chip>{conv.title}</Chip>}
      </header>

      <div className="scroll" ref={scroller} onScroll={onScroll}>
        <div className="col">
          {empty && (
            <>
              <Hero name={account ? displayName(account.email) : ""} tapes={!!source} compact={!source} />
              {source ? (
                <div className="starter">
                  <LanguageBridge config={config} locked videoLang={source.lang_code} answerLang={settings.answer_lang}
                    onAnswerLang={(v) => updateSettings({ answer_lang: v })} />
                  <div className="sug-cap"><Chip kind="acc">try asking</Chip></div>
                  <div className="sugs">{config.suggestions.map((s) => (
                    <button key={s} className="sug" onClick={() => ask(s)}><span>{s}</span><ArrowUp size={14} /></button>))}</div>
                  <p className="rib-note">Want another video or language? Open the Sources tab.</p>
                </div>
              ) : (
                <div className="onboard">
                  <ol className="steps3" aria-label="How it works">
                    <li><b>1</b><span>Paste a link</span></li>
                    <li><b>2</b><span>Janani listens</span></li>
                    <li><b>3</b><span>Ask anything</span></li>
                  </ol>
                  <AddVideo title="Start with a video" />
                  <Tapes words={TICKER} />
                </div>
              )}
            </>
          )}
          {msgs.map((m, i) => (
            <MessageView key={m.id} msg={m} isLast={i === msgs.length - 1 && m.role === 'assistant'} busy={!!live}
              config={config} showReasoning={settings.show_reasoning}
              onRetry={() => { const q = [...msgs].reverse().find((x) => x.role === 'user'); if (q) ask(q.content, true) }} />
          ))}
          <LiveAnswer />
        </div>
        {showJump && msgs.length > 0 && <button className="jump" onClick={() => scroller.current.scrollTo({ top: 1e9, behavior: 'smooth' })} aria-label="Scroll to latest"><ArrowDown size={16} /></button>}
      </div>
      {source ? <Composer disabled={false} /> : !empty && <Composer disabled reason="This conversation has no video. Pick or add one in Sources." />}
    </div>
  )
}
