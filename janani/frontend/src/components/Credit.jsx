import PvLogo from './PvLogo'

/** Maker credit with the PV mark. */
export default function Credit({ className = '' }) {
  return (
    <footer className={`credit ${className}`}>
      <PvLogo size={22} />
      <span>&copy; {new Date().getFullYear()} Made by <b>PV Sairam Saketh</b></span>
    </footer>
  )
}
