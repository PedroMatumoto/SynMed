import { Outlet } from 'react-router-dom'
import { AnalysisProvider } from '../contexts/analysisContext'

export function AnalysisWrapper() {
  return (
    <AnalysisProvider>
      <Outlet />
    </AnalysisProvider>
  )
}
