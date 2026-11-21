import { Fragment, useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Check, Eye, EyeOff, Languages, Loader2, Lock, Mail, MessageSquareQuote, ShieldCheck, Sparkles } from 'lucide-react'
import { api } from '../api'
import { displayName, fmtClock, isEmail, maskEmail, passwordStrength } from '../format'
import { Brand } from './TopBar'
import Stickers from './Stickers'

const LEN = 6
const DEAD = ['challenge_invalid', 'code_expired', 'too_many_attempts']
const LAST_EMAIL = 'janani-last-email'
const FEATURES = [
  [MessageSquareQuote, 'Grounded in the video'],
  [Languages, '20 languages'],
  [ShieldCheck, 'Private to you'],
]

const readLast = () => { try { return localStorage.getItem(LAST_EMAIL) || '' } catch { return '' } }
const saveLast = (v) => { try { localStorage.setItem(LAST_EMAIL, v) } catch { /* private mode */ } }

const SCENES = [
  { from: 'Telugu', to: 'English', q: 'బ్యాక్‌ప్రొపగేషన్ అంటే ఏమిటి?',
    a: 'It is how a network learns: it measures its error, then nudges every weight backwards to shrink it.', chips: ['#3 · 76%', '#11 · 73%'] },
  { from: 'Hindi', to: 'Spanish', q: 'ग्रेडिएंट डिसेंट क्या करता है?',
    a: 'Mueve los pesos poco a poco en la dirección que más reduce el error.', chips: ['#4 · 74%', '#1 · 71%'] },
  { from: 'Japanese', to: 'Hindi', q: 'コスト関数とは何ですか？',
    a: 'यह बताता है कि नेटवर्क का अनुमान सही उत्तर से कितना दूर है।', chips: ['#2 · 78%', '#7 · 70%'] },
  { from: 'English', to: 'French', q: 'What does the bias do?',
    a: 'Le biais décide à partir de quel seuil le neurone s’active.', chips: ['#5 · 75%', '#9 · 72%'] },
]
const graphemes = (t) => (typeof Intl !== 'undefined' && Intl.Segmenter
  ? Array.from(new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(t), (x) => x.segment) : Array.from(t))

/** A looping mini-conversation: question types in one language, the answer streams in another. */
function LiveDemo() {
  const reduce = useRef(typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches).current
  const [i, setI] = useState(0)
  const [phase, setPhase] = useState(reduce ? 'done' : 'q')      // q → think → a → done
  const [n, setN] = useState(0)
  const sc = SCENES[i]
  const qg = graphemes(sc.q)
  const words = sc.a.split(' ')
  useEffect(() => {
    if (reduce) return undefined
    let t
    if (phase === 'q') t = n < qg.length ? setTimeout(() => setN(n + 1), 48) : setTimeout(() => { setPhase('think'); setN(0) }, 380)
    else if (phase === 'think') t = setTimeout(() => setPhase('a'), 1150)
    else if (phase === 'a') t = n < words.length ? setTimeout(() => setN(n + 1), 105) : setTimeout(() => setPhase('done'), 300)
    else t = setTimeout(() => { setI((x) => (x + 1) % SCENES.length); setPhase('q'); setN(0) }, 3400)
    return () => clearTimeout(t)
  }, [reduce, phase, n, qg.length, words.length])
  const typedQ = phase === 'q' ? qg.slice(0, n).join('') : sc.q
  const shownA = phase === 'a' ? words.slice(0, n) : phase === 'done' ? words : []
  return (
    <div className="demo" aria-hidden="true">
      <div className="demo-bar">
        <i /><i /><i />
        <span>Janani · Backpropagation, intuitively</span>
      </div>
      <div className="demo-route" key={i}>
        <span>{sc.from}</span><b><ArrowRight size={13} /></b><span className="to">{sc.to}</span>
      </div>
      <div className="demo-body">
        {phase !== 'q' || n > 0 ? <div className="demo-q">{typedQ}{phase === 'q' && <u className="caret" />}</div> : <div className="demo-q ghost">…</div>}
        <div className="demo-a">
          <span className="demo-av"><Sparkles size={14} /></span>
          <div>
            {phase === 'think' && <div className="demo-think"><i /><i /><i /></div>}
            {shownA.length > 0 && <p>{shownA.map((w, k) => <Fragment key={k}><span className="w">{w}</span>{' '}</Fragment>)}</p>}
            <div className="demo-chips">
              {phase === 'done' && sc.chips.map((c, k) => <motion.span key={c} initial={{ opacity: 0, y: 6, scale: .9 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ delay: k * .12 }}>{c}</motion.span>)}
            </div>
          </div>
        </div>
      </div>
      <div className="demo-dots">{SCENES.map((_, k) => <i key={k} className={k === i ? 'on' : ''} />)}</div>
    </div>
  )
}

/** Left panel: the story, a live demo, and three quiet proof points. Hidden on small screens. */
function Showcase() {
  return (
    <aside className="showcase">
      <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .6, delay: .1 }}>
        <span className="showcase-pill"><Sparkles size={14} />Janani studio</span>
        <h2>Ask any video.<br /><em>In any language.</em></h2>
        <p>Paste a YouTube link, then talk to it. Janani listens, finds the exact passages, and answers in the language you choose.</p>
        <LiveDemo />
        <ul className="proofs">
          {FEATURES.map(([Icon, title], i) => (
            <motion.li key={title} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .5 + i * .1 }}>
              <span className="fi"><Icon size={15} /></span>{title}
            </motion.li>
          ))}
        </ul>
      </motion.div>
    </aside>
  )
}

/** Circular countdown for the code's remaining life. */
function Ring({ secs, total }) {
  const r = 15
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(1, secs / total))
  return (
    <svg className="ring" viewBox="0 0 36 36" aria-hidden="true">
      <circle cx="18" cy="18" r={r} className="ring-bg" />
      <circle cx="18" cy="18" r={r} className={`ring-fg ${pct < .2 ? 'low' : ''}`} strokeDasharray={c} strokeDashoffset={c * (1 - pct)} />
    </svg>
  )
}

/** Six boxes: type, paste, backspace and arrow keys all behave like a native one-time-code field. */
function CodeBoxes({ value, onChange, onComplete, disabled, shake }) {
  const refs = useRef([])
  useEffect(() => { refs.current[0]?.focus() }, [])
  const set = (i, ch) => {
    const next = [...value]
    next[i] = ch
    onChange(next)
    if (ch && i < LEN - 1) refs.current[i + 1]?.focus()
    if (next.every(Boolean)) onComplete(next.join(''))
  }
  const paste = (e) => {
    const digits = (e.clipboardData.getData('text') || '').replace(/\D/g, '').slice(0, LEN)
    if (!digits) return
    e.preventDefault()
    const next = Array.from({ length: LEN }, (_, i) => digits[i] || '')
    onChange(next)
    refs.current[Math.min(digits.length, LEN - 1)]?.focus()
    if (digits.length === LEN) onComplete(digits)
  }
  return (
    <motion.div className="otp" key={shake} animate={shake ? { x: [0, -10, 10, -7, 7, -3, 3, 0] } : {}} transition={{ duration: .45 }}>
      {Array.from({ length: LEN }, (_, i) => (
        <input key={i} ref={(el) => { refs.current[i] = el }} className={value[i] ? 'filled' : ''} inputMode="numeric"
          autoComplete={i === 0 ? 'one-time-code' : 'off'} maxLength={1} value={value[i] || ''} disabled={disabled}
          aria-label={`Digit ${i + 1} of ${LEN}`}
          onChange={(e) => { const d = e.target.value.replace(/\D/g, '').slice(-1); if (d || e.target.value === '') set(i, d) }}
          onKeyDown={(e) => {
            if (e.key === 'Backspace' && !value[i] && i > 0) { refs.current[i - 1]?.focus() }
            if (e.key === 'ArrowLeft' && i > 0) refs.current[i - 1]?.focus()
            if (e.key === 'ArrowRight' && i < LEN - 1) refs.current[i + 1]?.focus()
          }}
          onPaste={paste} onFocus={(e) => e.target.select()} />
      ))}
    </motion.div>
  )
}

export default function Login({ onDone }) {
  const [step, setStep] = useState('creds')          // creds | code | done
  const [email, setEmail] = useState(readLast)
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [caps, setCaps] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [ch, setCh] = useState(null)                   // { token, masked, purpose, resendAt, expiresAt, total }
  const [code, setCode] = useState(Array(LEN).fill(''))
  const [shake, setShake] = useState(0)
  const [now, setNow] = useState(Date.now())
  const [mail, setMail] = useState(null)             // server's email setup, from /api/health/ (never secrets)
  const mounted = useRef(true)

  // set true on every mount: React StrictMode (dev) mounts, unmounts and re-mounts, and the flag must survive that
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t) }, [])
  useEffect(() => { api.get('/health/').then((d) => setMail(d.email || null)).catch(() => {}) }, [])

  const strength = passwordStrength(password)
  const resendIn = ch ? Math.max(0, Math.ceil((ch.resendAt - now) / 1000)) : 0
  const expiresIn = ch ? Math.max(0, Math.ceil((ch.expiresAt - now) / 1000)) : 0
  const name = displayName(email)

  async function start(e) {
    e?.preventDefault()
    if (busy) return
    setError('')
    if (!isEmail(email)) { setError('Enter a valid email address.'); return }
    if (!password) { setError('Enter your password.'); return }
    setBusy(true)
    try {
      const r = await api.post('/auth/start/', { email: email.trim(), password })
      saveLast(email.trim().toLowerCase())
      setCh({ token: r.challenge_token, masked: r.masked, purpose: r.purpose, resendAt: Date.now() + r.resend_in * 1000, expiresAt: Date.now() + r.expires_in * 1000, total: r.expires_in })
      setCode(Array(LEN).fill('')); setStep('code')
    } catch (err) {
      if (err.code === 'cooldown' && err.extra.challenge_token) {      // a code was just sent: go straight to entering it
        setCh({ token: err.extra.challenge_token, masked: maskEmail(email), purpose: 'login', resendAt: Date.now() + (err.extra.retry_after || 45) * 1000, expiresAt: Date.now() + 9 * 60 * 1000, total: 600 })
        setCode(Array(LEN).fill('')); setStep('code')
      } else setError(err.message)
    } finally { if (mounted.current) setBusy(false) }
  }

  const verify = useCallback(async (digits) => {
    if (busy || !ch) return
    setBusy(true); setError('')
    try {
      await api.post('/auth/verify/', { challenge_token: ch.token, code: digits })
      setStep('done')
      setTimeout(() => onDone?.(), 1100)
    } catch (err) {
      setError(err.message); setShake((s) => s + 1); setCode(Array(LEN).fill(''))
      if (DEAD.includes(err.code)) setCh((c) => ({ ...c, dead: true }))
    } finally { if (mounted.current) setBusy(false) }
  }, [busy, ch, onDone])

  async function resend() {
    if (busy || resendIn > 0 || !ch) return
    setBusy(true); setError('')
    try {
      const r = await api.post('/auth/resend/', { challenge_token: ch.token })
      setCh({ ...ch, token: r.challenge_token, dead: false, resendAt: Date.now() + r.resend_in * 1000, expiresAt: Date.now() + r.expires_in * 1000, total: r.expires_in })
      setCode(Array(LEN).fill(''))
    } catch (err) {
      setError(err.message)
      if (err.code === 'challenge_invalid') setCh((c) => ({ ...c, dead: true }))
    } finally { if (mounted.current) setBusy(false) }
  }

  const back = () => { setStep('creds'); setError(''); setCh(null); setCode(Array(LEN).fill('')) }

  return (
    <div className="login-page">
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <div className="grain" aria-hidden />
      <Stickers />
      <div className="login-shell">
        <Showcase />
        <div className="login-side">
          <motion.div className="login-card" initial={{ opacity: 0, y: 24, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: .5, ease: [.2, .8, .2, 1] }}>
            <div className="login-brand"><Brand /></div>
            <div className="steps" aria-hidden="true">
              <i className={step === 'creds' ? 'on' : 'done'} /><i className={step === 'code' ? 'on' : step === 'done' ? 'done' : ''} /><i className={step === 'done' ? 'on' : ''} />
            </div>
            <AnimatePresence mode="wait">
              {step === 'creds' && (
                <motion.form key="creds" onSubmit={start} noValidate initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -16 }} transition={{ duration: .25 }}>
                  <h1>{email && readLast() && isEmail(email) ? <>Welcome <em>back</em></> : <>Welcome <em>in</em></>}</h1>
                  <p className="login-sub">Sign in or create your account. We’ll email you a 6-digit code to confirm it’s you.</p>
                  <label className="field"><span>Email</span>
                    <div className="field-box"><Mail size={17} />
                      <input type="email" autoComplete="username" autoFocus={!email} value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" disabled={busy} spellCheck={false} />
                    </div>
                  </label>
                  <label className="field"><span>Password</span>
                    <div className="field-box"><Lock size={17} />
                      <input type={show ? 'text' : 'password'} autoComplete="current-password" autoFocus={!!email} value={password}
                        onChange={(e) => setPassword(e.target.value)} placeholder="At least 8 characters" disabled={busy}
                        onKeyUp={(e) => setCaps(e.getModifierState?.('CapsLock') || false)} onBlur={() => setCaps(false)} />
                      <button type="button" className="eye" onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide password' : 'Show password'}>{show ? <EyeOff size={17} /> : <Eye size={17} />}</button>
                    </div>
                  </label>
                  {caps && <p className="login-caps">Caps Lock is on</p>}
                  {password.length > 0 && <div className="meter" aria-hidden><i style={{ width: `${(strength / 4) * 100}%` }} className={`s${strength}`} /></div>}
                  <p className="login-hint">New here? Pick a password (8+ characters). Your account is created once you confirm the code.</p>
                  {mail && mail.configured === false && (
                    <p className="login-warn" role="status">Email sign-in isn’t set up on this server yet, so codes can’t be sent. The owner needs to finish the email setup (see the README).</p>
                  )}
                  {error && <p className="login-err" role="alert">{error}</p>}
                  <button className="btn primary wide big" disabled={busy}>
                    {busy ? <Loader2 size={17} className="spin" /> : <ArrowRight size={17} />}{busy ? 'Sending your code…' : 'Continue'}
                  </button>
                </motion.form>
              )}

              {step === 'code' && (
                <motion.div key="code" initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 16 }} transition={{ duration: .25 }}>
                  <div className="login-icon"><Mail size={22} /></div>
                  <h1>Check your <em>inbox</em></h1>
                  <p className="login-sub">We sent a 6-digit code to <b>{ch.masked}</b>{ch.purpose === 'signup' ? ' to create your account.' : '.'}</p>
                  <CodeBoxes value={code} onChange={setCode} onComplete={verify} disabled={busy || ch.dead} shake={shake} />
                  {error && <p className="login-err" role="alert">{error}</p>}
                  <div className="login-row">
                    <span className="expiry">
                      <Ring secs={expiresIn} total={ch.total || 600} />
                      {ch.dead ? 'Code can’t be used any more' : expiresIn > 0 ? <>Expires in <b>{fmtClock(expiresIn)}</b></> : 'Code expired'}
                    </span>
                    <button type="button" className="linklike" onClick={resend} disabled={busy || resendIn > 0}>
                      {resendIn > 0 ? `Resend in ${fmtClock(resendIn)}` : 'Send a new code'}
                    </button>
                  </div>
                  <button type="button" className="btn primary wide big" disabled={busy || code.some((c) => !c)} onClick={() => verify(code.join(''))}>
                    {busy ? <Loader2 size={17} className="spin" /> : <Check size={17} />}Verify and continue
                  </button>
                  <button type="button" className="login-back" onClick={back}><ArrowLeft size={15} />Use a different email</button>
                  <p className="login-hint center">It comes from <b>pvsaketh1@gmail.com</b>. Can’t find it? Check your spam folder.</p>
                </motion.div>
              )}

              {step === 'done' && (
                <motion.div key="done" className="login-done" initial={{ opacity: 0, scale: .9 }} animate={{ opacity: 1, scale: 1 }}>
                  <div className="confetti" aria-hidden>{Array.from({ length: 18 }, (_, i) => <i key={i} style={{ '--a': `${i * 20}deg`, '--d': `${70 + (i % 4) * 18}px`, '--c': ['#5b5bf0', '#c04cf0', '#ff7a59', '#22c1a0', '#ffd36e'][i % 5] }} />)}</div>
                  <motion.div className="login-check" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 18 }}><Check size={34} strokeWidth={3} /></motion.div>
                  <h1>Welcome, <em>{name}</em></h1>
                  <p className="login-sub">Opening your workspace…</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
          <p className="login-foot">Your chats and videos are private to your account.</p>
        </div>
      </div>
    </div>
  )
}
