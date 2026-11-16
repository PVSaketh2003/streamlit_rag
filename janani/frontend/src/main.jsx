import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { AppProvider } from './store'
import { ErrorBoundary } from './components/ui'
import App from './App'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <StrictMode><ErrorBoundary><AppProvider><App /></AppProvider></ErrorBoundary></StrictMode>
)
