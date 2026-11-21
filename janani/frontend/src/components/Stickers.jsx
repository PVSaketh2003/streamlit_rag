/** Decorative paper stickers that float near the edges. Purely visual; hidden on small screens. */
export default function Stickers() {
  const ink = '#14121f'
  return (
    <div className="stickers" aria-hidden="true">
      <svg className="stk s1" viewBox="0 0 100 100"><path d="M50 4l11 27 29-5-19 22 19 22-29-5-11 27-11-27-29 5 19-22-19-22 29 5z" fill="#ffd84d" stroke={ink} strokeWidth="3" strokeLinejoin="round" /></svg>
      <svg className="stk s2" viewBox="0 0 120 40"><path d="M4 20c10-18 20-18 30 0s20 18 30 0 20-18 30 0 14 12 22 4" fill="none" stroke={ink} strokeWidth="4" strokeLinecap="round" /></svg>
      <svg className="stk s3" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="#ffc6e0" stroke={ink} strokeWidth="3" /><circle cx="50" cy="50" r="18" fill="#fff" stroke={ink} strokeWidth="3" /></svg>
      <svg className="stk s4" viewBox="0 0 100 100"><path d="M38 6h24v32h32v24H62v32H38V62H6V38h32z" fill="#bff0d4" stroke={ink} strokeWidth="3" strokeLinejoin="round" /></svg>
      <svg className="stk s5" viewBox="0 0 100 60"><path d="M4 56a46 46 0 0192 0z" fill="#d9ccff" stroke={ink} strokeWidth="3" strokeLinejoin="round" /></svg>
    </div>
  )
}
