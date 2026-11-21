/** "PV" monogram: the maker's mark. Ink-outlined sticker with a tiny spark. */
export default function PvLogo({ size = 26, className = '' }) {
  return (
    <svg className={`pvlogo ${className}`} viewBox="0 0 64 64" width={size} height={size} role="img" aria-label="PV">
      <rect x="4" y="4" width="56" height="56" rx="16" fill="#ff9ecb" stroke="#14121f" strokeWidth="4" />
      <text x="32" y="43" textAnchor="middle" fontFamily="'Bricolage Grotesque', 'Inter', system-ui, sans-serif" fontWeight="800" fontSize="29" letterSpacing="-2" fill="#14121f">PV</text>
      <path d="M52 8l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#ffd84d" stroke="#14121f" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  )
}
