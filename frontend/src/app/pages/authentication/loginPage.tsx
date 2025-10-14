import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/app/contexts/authContext'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPills, faUser, faArrowLeft, faEye, faEyeSlash } from '@fortawesome/free-solid-svg-icons'
import AuroraBackground from '@/app/components/ui/auroraBackground'

export default function LoginPage() {
  const { login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      await login(email, password)
      navigate('/analysis')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao fazer login')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuroraBackground className="min-h-screen w-screen">
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="w-full max-w-md">
          {/* Back to Home Button */}
          <button
            onClick={() => navigate('/')}
            className="mb-6 flex items-center gap-2 text-white/70 hover:text-white transition-colors"
          >
            <FontAwesomeIcon icon={faArrowLeft} />
            Voltar ao início
          </button>

          {/* Login Card */}
          <div className="animate-fadeIn rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 p-8 shadow-2xl">
            {/* Header */}
            <div className="mb-8 text-center">
              <div className="mb-4 flex items-center justify-center gap-3">
                <div className="rounded-xl bg-accent/20 p-3">
                  <FontAwesomeIcon icon={faPills} className="text-2xl text-white" />
                </div>
                <h1 className="text-2xl font-bold text-white">SynMed</h1>
              </div>
              <p className="text-dark-300">
                Entre com sua conta para continuar
              </p>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
              {error && (
                <div className="rounded-lg bg-red-500/20 border border-red-500/30 p-3 text-center text-red-200">
                  {error}
                </div>
              )}

              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium text-dark-200">
                  E-mail
                </label>
                <input
                  onChange={(e) => setEmail(e.target.value)}
                  value={email}
                  type="email"
                  id="email"
                  className="w-full rounded-lg bg-white/5 border border-white/20 px-4 py-3 text-white placeholder-dark-400 outline-none transition-all focus:border-accent focus:bg-white/10"
                  placeholder="seu@email.com"
                  required
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="password" className="text-sm font-medium text-dark-200">
                  Senha
                </label>
                <div className="relative">
                  <input
                    onChange={(e) => setPassword(e.target.value)}
                    value={password}
                    type={showPassword ? 'text' : 'password'}
                    id="password"
                    className="w-full rounded-lg bg-white/5 border border-white/20 px-4 py-3 pr-12 text-white placeholder-dark-400 outline-none transition-all focus:border-accent focus:bg-white/10"
                    placeholder="Sua senha"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-dark-400 hover:text-white transition-colors"
                  >
                    <FontAwesomeIcon icon={showPassword ? faEyeSlash : faEye} />
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="group w-full flex items-center justify-center gap-3 rounded-lg bg-accent px-6 py-3 font-semibold text-white transition-all duration-300 hover:bg-accent/90 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {loading ? (
                    <>
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                      Entrando...
                    </>
                  ) : (
                    <>
                      Entrar
                      <FontAwesomeIcon icon={faPills} className="transition-transform group-hover:rotate-12" />
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => navigate('/register')}
                  className="group w-full flex items-center justify-center gap-3 rounded-lg border border-white/20 bg-white/5 px-6 py-3 font-semibold text-white transition-all duration-300 hover:bg-white/10 hover:border-white/30"
                >
                  Criar nova conta
                  <FontAwesomeIcon icon={faUser} className="transition-transform group-hover:scale-110" />
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </AuroraBackground>
  )
}
