/** Animated sound-wave: "Janani is listening". Pure CSS, respects reduced motion. */
import Mascot from './Mascot'

export default function Wave({ bars = 36, label, mascot = false }) {
  return (
    <div className="listen" role="status" aria-label={label || 'Listening'}>
      {mascot && <Mascot size={84} mood="listen" />}
      <div className="wave" aria-hidden="true">{Array.from({ length: bars }, (_, i) => <i key={i} style={{ '--i': i }} />)}</div>
      {label && <span>{label}</span>}
    </div>
  )
}
