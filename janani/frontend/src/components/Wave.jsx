/** Animated sound-wave: "Janani is listening". Pure CSS, respects reduced motion. */
export default function Wave({ bars = 36, label }) {
  return (
    <div className="listen" role="status" aria-label={label || 'Listening'}>
      <div className="wave" aria-hidden="true">{Array.from({ length: bars }, (_, i) => <i key={i} style={{ '--i': i }} />)}</div>
      {label && <span>{label}</span>}
    </div>
  )
}
