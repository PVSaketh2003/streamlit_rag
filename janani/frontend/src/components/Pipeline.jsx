import { motion } from 'framer-motion'
import { Check, Loader2, X } from 'lucide-react'
import { fmtMs } from '../format'

/** Horizontal stage ribbon. `stages` = [{key,label}], `state` = {key:{status,ms,detail}}. */
export default function Pipeline({ stages, state, compact }) {
  return (
    <ol className={`pipe ${compact ? 'compact' : ''}`}>
      {stages.map((s, i) => {
        const st = state[s.key] || { status: 'idle' }
        return (
          <li key={s.key} className={`pipe-step ${st.status}`} title={st.detail || ''}>
            <span className="pipe-dot">
              {st.status === 'done' ? <Check size={12} strokeWidth={3} />
                : st.status === 'running' ? <Loader2 size={12} className="spin" />
                : st.status === 'error' ? <X size={12} strokeWidth={3} /> : i + 1}
            </span>
            <span className="pipe-body">
              <b>{s.label}</b>
              <small>{st.status === 'done' ? fmtMs(st.ms) : st.status === 'running' ? (st.detail || 'working…') : st.status === 'error' ? 'failed' : '·'}</small>
            </span>
            {i < stages.length - 1 && (
              <span className="pipe-line"><motion.i initial={false} animate={{ scaleX: st.status === 'done' ? 1 : 0 }}
                transition={{ duration: .4 }} /></span>
            )}
          </li>
        )
      })}
    </ol>
  )
}
