import { useEffect } from 'react'

const SPOT = '.panel, .sug, .tile, .addvideo, .chunk, .how, .src, .bridge, .preview, .composer, .demo, .login-card, .proofs li, .ai-body, .empty-live, .steps3 li'
const TILT = '.sug, .src'
const MAGNET = '.btn.primary, .send, .dock button, .proofs li'

/** Pointer effects via event delegation: card spotlight, 3D tilt, magnetic buttons, cursor halo. */
export function useFx() {
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || matchMedia('(pointer: coarse)').matches) return
    const root = document.documentElement
    let raf = 0, ev = null, lastMag = null, lastTilt = null
    const INTERACTIVE = 'button, a, input, textarea, select, summary, [role="switch"], label'
    const reset = (el) => { if (el) { el.style.transform = ''; el.style.removeProperty('--rx'); el.style.removeProperty('--ry') } }
    const frame = () => {
      raf = 0
      if (!ev) return
      const { clientX: x, clientY: y, target } = ev
      root.style.setProperty('--cx', `${x}px`); root.style.setProperty('--cy', `${y}px`)
      const el = target instanceof Element ? target : null
      const spot = el?.closest(SPOT)
      if (spot) {
        const r = spot.getBoundingClientRect()
        spot.style.setProperty('--mx', `${x - r.left}px`); spot.style.setProperty('--my', `${y - r.top}px`)
      }
      const tilt = el?.closest(TILT)
      if (lastTilt && lastTilt !== tilt) reset(lastTilt)
      if (tilt) {
        const r = tilt.getBoundingClientRect()
        const px = (x - r.left) / r.width - .5, py = (y - r.top) / r.height - .5
        tilt.style.transform = `perspective(700px) rotateX(${(-py * 5).toFixed(2)}deg) rotateY(${(px * 6).toFixed(2)}deg) translateZ(0)`
      }
      lastTilt = tilt
      const mag = el?.closest(MAGNET)
      if (lastMag && lastMag !== mag) reset(lastMag)
      if (mag && !mag.disabled) {
        const r = mag.getBoundingClientRect()
        const dx = (x - (r.left + r.width / 2)) * .22, dy = (y - (r.top + r.height / 2)) * .28
        mag.style.transform = `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`
      }
      lastMag = mag
      const ring = document.querySelector('.cursor-ring')
      if (ring) {
        ring.style.transform = `translate3d(${x}px, ${y}px, 0)`
        ring.classList.add('show')
        ring.classList.toggle('big', !!el?.closest(INTERACTIVE))
        ring.classList.toggle('text', !!el?.closest('textarea, input[type="text"], input:not([type])'))
      }
      // hero orb parallax
      const orb = document.querySelector('.orb')
      if (orb) {
        const r = orb.getBoundingClientRect()
        const ox = Math.max(-1, Math.min(1, (x - (r.left + r.width / 2)) / 400))
        const oy = Math.max(-1, Math.min(1, (y - (r.top + r.height / 2)) / 400))
        orb.style.setProperty('--ox', ox.toFixed(3)); orb.style.setProperty('--oy', oy.toFixed(3))
      }
    }
    // sparkle trail over the welcome / sign-in surfaces, and a shockwave ring on every press
    const COLORS = ['#5b5bf0', '#c04cf0', '#ff7a59', '#ffd36e', '#22c1a0']
    let lastSpark = 0, sparks = 0
    const spark = (x, y) => {
      if (sparks > 24) return
      sparks++
      const d = document.createElement('i')
      d.className = 'spark'
      const a = Math.random() * Math.PI * 2, r = 14 + Math.random() * 26
      d.style.cssText = `left:${x}px;top:${y}px;--dx:${(Math.cos(a) * r).toFixed(1)}px;--dy:${(Math.sin(a) * r + 14).toFixed(1)}px;--c:${COLORS[(Math.random() * COLORS.length) | 0]};--s:${(6 + Math.random() * 8).toFixed(1)}px`
      document.body.appendChild(d)
      setTimeout(() => { d.remove(); sparks-- }, 900)
    }
    const shock = (e) => {
      if (!(e.target instanceof Element) || !e.target.closest('.btn, .send, .sug, .dock button, .login-card button, .src, .chip')) return
      const d = document.createElement('i')
      d.className = 'shock'
      d.style.cssText = `left:${e.clientX}px;top:${e.clientY}px`
      document.body.appendChild(d)
      setTimeout(() => d.remove(), 700)
    }
    const move = (e) => {
      ev = e
      const now = performance.now()
      if (now - lastSpark > 55 && e.target instanceof Element && e.target.closest('.hero, .login-page, .onboard') && !e.target.closest('input, textarea, button, select')) {
        lastSpark = now; spark(e.clientX, e.clientY)
      }
      if (!raf) raf = requestAnimationFrame(frame)
    }
    const leave = () => {
      reset(lastMag); reset(lastTilt); lastMag = lastTilt = null
      document.querySelector('.cursor-ring')?.classList.remove('show')
    }
    const down = () => document.querySelector('.cursor-ring')?.classList.add('down')
    const up = () => document.querySelector('.cursor-ring')?.classList.remove('down')
    window.addEventListener('pointerdown', down)
    window.addEventListener('pointerdown', shock)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointermove', move, { passive: true })
    document.addEventListener('pointerleave', leave)
    return () => {
      window.removeEventListener('pointermove', move); document.removeEventListener('pointerleave', leave)
      window.removeEventListener('pointerdown', down); window.removeEventListener('pointerdown', shock); window.removeEventListener('pointerup', up); cancelAnimationFrame(raf)
    }
  }, [])
}
