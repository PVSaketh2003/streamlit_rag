import { useEffect } from 'react'

const SPOT = '.panel, .sug, .tile, .addvideo, .chunk, .how, .src, .bridge, .preview, .composer'
const TILT = '.sug, .src'
const MAGNET = '.btn.primary, .send, .dock button'

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
    const move = (e) => { ev = e; if (!raf) raf = requestAnimationFrame(frame) }
    const leave = () => {
      reset(lastMag); reset(lastTilt); lastMag = lastTilt = null
      document.querySelector('.cursor-ring')?.classList.remove('show')
    }
    const down = () => document.querySelector('.cursor-ring')?.classList.add('down')
    const up = () => document.querySelector('.cursor-ring')?.classList.remove('down')
    window.addEventListener('pointerdown', down)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointermove', move, { passive: true })
    document.addEventListener('pointerleave', leave)
    return () => {
      window.removeEventListener('pointermove', move); document.removeEventListener('pointerleave', leave)
      window.removeEventListener('pointerdown', down); window.removeEventListener('pointerup', up); cancelAnimationFrame(raf)
    }
  }, [])
}
