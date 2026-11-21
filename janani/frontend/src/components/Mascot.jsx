import { useEffect, useRef, useState } from 'react'

const INK = '#14121f'

/** Janani, the mascot. Eyes follow the cursor, blinks, wears headphones while listening, thinks while answering.
 *  mood: idle | listen | think | happy. Click for a hop. Decorative (hidden from screen readers). */
export default function Mascot({ size = 120, mood = 'idle', className = '', style }) {
  const svg = useRef(null)
  const pup = useRef(null)
  const [hop, setHop] = useState(false)
  const m = hop ? 'happy' : mood

  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || matchMedia('(pointer: coarse)').matches) return undefined
    let raf = 0, ev = null
    const frame = () => {
      raf = 0
      const el = svg.current, p = pup.current
      if (!el || !p || !ev) return
      const r = el.getBoundingClientRect()
      const dx = ev.clientX - (r.left + r.width / 2), dy = ev.clientY - (r.top + r.height * .5)
      const d = Math.hypot(dx, dy) || 1, k = Math.min(1, d / 200)
      p.style.transform = `translate(${((dx / d) * 6 * k).toFixed(2)}px, ${((dy / d) * 6 * k).toFixed(2)}px)`
    }
    const move = (e) => { ev = e; if (!raf) raf = requestAnimationFrame(frame) }
    window.addEventListener('pointermove', move, { passive: true })
    return () => { window.removeEventListener('pointermove', move); cancelAnimationFrame(raf) }
  }, [])

  const poke = () => { setHop(true); setTimeout(() => setHop(false), 1300) }
  const thinkLook = m === 'think' ? { transform: 'translate(3px, -6px)' } : undefined

  return (
    <svg ref={svg} viewBox="0 0 200 210" width={size} height={Math.round(size * 1.05)} className={`mascot m-${m} ${hop ? 'hop' : ''} ${className}`}
      style={style} aria-hidden="true" onClick={poke}>
      <ellipse className="m-shadow" cx="100" cy="198" rx="54" ry="8" fill="rgba(20,18,31,.2)" />
      <g className="m-body">
        <line x1="104" y1="36" x2="117" y2="12" stroke={INK} strokeWidth="5" strokeLinecap="round" />
        <circle className="m-tip" cx="119" cy="10" r="9" fill="#ff7ab8" stroke={INK} strokeWidth="4" />
        <path d="M18 132c-12-6-14-24-2-30 10-5 20 4 22 16" fill="#ffd84d" stroke={INK} strokeWidth="5" strokeLinecap="round" className="m-arm l" />
        <path d="M182 132c12-6 14-24 2-30-10-5-20 4-22 16" fill="#ffd84d" stroke={INK} strokeWidth="5" strokeLinecap="round" className="m-arm r" />
        <path d="M38 126C36 66 74 32 106 34c42 2 66 40 62 88-3 44-30 64-68 64-38 0-60-20-62-60z" fill="#ffd84d" stroke={INK} strokeWidth="5" strokeLinejoin="round" />
        <path d="M62 80c6-16 20-26 34-28" stroke="#fff" strokeWidth="6" strokeLinecap="round" fill="none" opacity=".75" />
        {m === 'listen' && (
          <g>
            <path d="M44 112C38 50 162 42 164 108" fill="none" stroke={INK} strokeWidth="7" strokeLinecap="round" />
            <rect x="28" y="98" width="20" height="42" rx="9" fill="#ff7ab8" stroke={INK} strokeWidth="4" />
            <rect x="158" y="98" width="20" height="42" rx="9" fill="#ff7ab8" stroke={INK} strokeWidth="4" />
          </g>
        )}
        <g className="m-eyes">
          <ellipse cx="80" cy="106" rx="17" ry="19" fill="#fff" stroke={INK} strokeWidth="4" />
          <ellipse cx="126" cy="106" rx="17" ry="19" fill="#fff" stroke={INK} strokeWidth="4" />
          <g ref={pup} className="m-pupils" style={thinkLook}>
            <circle cx="80" cy="108" r="8" fill={INK} /><circle cx="126" cy="108" r="8" fill={INK} />
            <circle cx="83" cy="104" r="2.6" fill="#fff" /><circle cx="129" cy="104" r="2.6" fill="#fff" />
          </g>
        </g>
        <circle cx="56" cy="138" r="9" fill="#ff9ecb" opacity=".85" /><circle cx="150" cy="138" r="9" fill="#ff9ecb" opacity=".85" />
        {m === 'idle' && <path d="M88 141q15 16 30 0" fill="none" stroke={INK} strokeWidth="5" strokeLinecap="round" />}
        {m === 'listen' && <path d="M92 144q11 8 22 0" fill="none" stroke={INK} strokeWidth="5" strokeLinecap="round" />}
        {m === 'think' && <ellipse className="m-o" cx="103" cy="148" rx="7" ry="8" fill={INK} />}
        {m === 'happy' && (<g><path d="M84 139q19 34 38 0z" fill={INK} stroke={INK} strokeWidth="4" strokeLinejoin="round" /><path d="M95 152q9 7 18 0" fill="#ff7ab8" /></g>)}
      </g>
      {m === 'think' && (<g className="m-dots"><circle cx="150" cy="40" r="5" fill={INK} /><circle cx="166" cy="28" r="6.5" fill={INK} /><circle cx="186" cy="14" r="8" fill={INK} /></g>)}
    </svg>
  )
}
