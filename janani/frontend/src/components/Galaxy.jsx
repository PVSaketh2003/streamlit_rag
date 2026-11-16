import { useEffect, useRef } from 'react'

/** Greetings in many scripts orbiting the orb in 3D. Billboarded spans positioned per frame:
 *  depth drives scale, opacity, blur and whether a word passes in front of or behind the orb. */
export default function Galaxy({ words }) {
  const ref = useRef(null)
  useEffect(() => {
    const root = ref.current
    if (!root) return
    const els = [...root.querySelectorAll('.gw')]
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
    let angle = 0, raf = 0, last = performance.now(), px = 0, py = 0, tx = 0, ty = 0
    const onMove = (e) => {
      const r = root.getBoundingClientRect()
      px = Math.max(-1, Math.min(1, (e.clientX - (r.left + r.width / 2)) / (r.width / 2)))
      py = Math.max(-1, Math.min(1, (e.clientY - (r.top + r.height / 2)) / 300))
    }
    const place = () => {
      const W = root.clientWidth / 2 * .82
      const H = 74 + ty * 26
      els.forEach((el, i) => {
        const th = angle + (i / els.length) * Math.PI * 2
        const z = Math.cos(th)
        const depth = (z + 1) / 2
        const x = Math.sin(th) * W
        const y = z * H
        el.style.transform = `translate(-50%, -50%) translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) scale(${(.6 + depth * .58).toFixed(3)})`
        el.style.opacity = (.14 + depth * .86).toFixed(3)
        el.style.filter = depth < .5 ? `blur(${((.5 - depth) * 5).toFixed(1)}px)` : 'none'
        el.style.zIndex = z > 0 ? 3 : 1
      })
    }
    const frame = (t) => {
      const dt = Math.min(48, t - last); last = t
      tx += (px - tx) * .04; ty += (py - ty) * .04
      angle += dt * (.00016 + tx * .0005)
      place()
      raf = requestAnimationFrame(frame)
    }
    place()
    if (!reduce) {
      window.addEventListener('pointermove', onMove, { passive: true })
      raf = requestAnimationFrame(frame)
    }
    return () => { cancelAnimationFrame(raf); window.removeEventListener('pointermove', onMove) }
  }, [words])
  return (
    <div className="galaxy" ref={ref} aria-hidden>
      {words.slice(0, 12).map((w, i) => <span key={i} className={`gw ${i % 4 === 0 ? 'g' : ''}`}>{w}</span>)}
      <svg className="goo-defs" aria-hidden><filter id="goo"><feGaussianBlur in="SourceGraphic" stdDeviation="14" result="b" />
        <feColorMatrix in="b" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 24 -10" /></filter></svg>
      <div className="lava"><b /><b /><b /><b /><b /></div>
      <div className="orb"><i /><i /><i /><u /></div>
    </div>
  )
}
