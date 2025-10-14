import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './pages/homepage'
import LoginPage from './pages/authentication/loginPage'
import RegisterPage from './pages/authentication/registerPage'
import AnalysisPage from './pages/analysisPage'
import HistoryPage from './pages/historyPage'
import { AuthWrapper } from './layouts/authWrapper'
import { AnalysisWrapper } from './layouts/analysisWrapper'

export function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route element={<AuthWrapper />}>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          <Route path="/analysis" element={<AnalysisWrapper />}>
            <Route index element={<AnalysisPage />} />
          </Route>

          <Route path="/history" element={<AnalysisWrapper />}>
            <Route index element={<HistoryPage />} />
          </Route>
        </Route>

        <Route path="*" element={<h1>Página não encontrada</h1>} />
      </Routes>
    </Router>
  )
}
