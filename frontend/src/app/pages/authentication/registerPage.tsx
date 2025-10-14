import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/app/contexts/authContext'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPills, faUser, faArrowLeft, faEye, faEyeSlash, faCheck } from '@fortawesome/free-solid-svg-icons'
import AuroraBackground from '@/app/components/ui/auroraBackground'

export default function RegisterPage() {
  const { register } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  // Password validation
  const passwordValidation = {
    minLength: password.length >= 6,
    hasNumber: /\d/.test(password),
    hasSpecial: /[!@#$%^&*(),.?":{}|<>]/.test(password)
  }

  const isPasswordValid = Object.values(passwordValidation).every(Boolean)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')

    if (!isPasswordValid) {
      setError('A senha deve atender a todos os critérios de segurança')
      return
    }

    setLoading(true)

    try {
      await register(fullName, email, password)
      navigate('/analysis')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao registrar')
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

          {/* Register Card */}
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
                Crie sua conta para começar suas análises
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
                <label htmlFor="full-name" className="text-sm font-medium text-dark-200">
                  Nome completo
                </label>
                <input
                  onChange={(e) => setFullName(e.target.value)}
                  value={fullName}
                  type="text"
                  id="full-name"
                  className="w-full rounded-lg bg-white/5 border border-white/20 px-4 py-3 text-white placeholder-dark-400 outline-none transition-all focus:border-accent focus:bg-white/10"
                  placeholder="Nome Sobrenome"
                  required
                />
              </div>

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
                    placeholder="Crie uma senha segura"
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

                {/* Password Validation */}
                {password && (
                  <div className="mt-3 space-y-2 text-xs">
                    <div className={`flex items-center gap-2 ${passwordValidation.minLength ? 'text-green-400' : 'text-red-400'}`}>
                      <FontAwesomeIcon icon={faCheck} className={passwordValidation.minLength ? 'opacity-100' : 'opacity-30'} />
                      Mínimo 6 caracteres
                    </div>
                    <div className={`flex items-center gap-2 ${passwordValidation.hasNumber ? 'text-green-400' : 'text-red-400'}`}>
                      <FontAwesomeIcon icon={faCheck} className={passwordValidation.hasNumber ? 'opacity-100' : 'opacity-30'} />
                      Pelo menos 1 número
                    </div>
                    <div className={`flex items-center gap-2 ${passwordValidation.hasSpecial ? 'text-green-400' : 'text-red-400'}`}>
                      <FontAwesomeIcon icon={faCheck} className={passwordValidation.hasSpecial ? 'opacity-100' : 'opacity-30'} />
                      Pelo menos 1 caractere especial
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <button
                  type="submit"
                  disabled={loading || !isPasswordValid}
                  className="group w-full flex items-center justify-center gap-3 rounded-lg bg-accent px-6 py-3 font-semibold text-white transition-all duration-300 hover:bg-accent/90 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {loading ? (
                    <>
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                      Criando conta...
                    </>
                  ) : (
                    <>
                      Criar conta
                      <FontAwesomeIcon icon={faPills} className="transition-transform group-hover:rotate-12" />
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => navigate('/login')}
                  className="group w-full flex items-center justify-center gap-3 rounded-lg border border-white/20 bg-white/5 px-6 py-3 font-semibold text-white transition-all duration-300 hover:bg-white/10 hover:border-white/30"
                >
                  Já tenho uma conta
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
