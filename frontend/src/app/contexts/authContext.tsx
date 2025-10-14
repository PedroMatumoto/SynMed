import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { User, Token } from '../types/user'
import { environments } from '@/utils/env/environments'

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<void>
  register: (full_name: string, email: string, password: string) => Promise<void>
  logout: () => void
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    const publicRoutes = ['/login', '/register', '/']

    if (publicRoutes.includes(location.pathname)) {
      setLoading(false)
      return
    }

    async function validateUser() {
      const storedUser = localStorage.getItem('user')
      if (storedUser) {
        const parsedUser = JSON.parse(storedUser)
        setUser(parsedUser)

        try {
          await me(parsedUser.access_token)
        } catch (error) {
          // Token inválido, remove e limpa usuário
          localStorage.removeItem('user')
          setUser(null)
          navigate('/login')
        }
      } else {
        navigate('/login')
      }
      setLoading(false)
    }

    validateUser()
  }, [location.pathname, navigate])

  async function me(access_token: string) {
    const meRes = await fetch(`${environments.apiUrl}/auth/me`, {
      headers: { Authorization: `Bearer ${access_token}` }
    })

    if (!meRes.ok) {
      throw new Error('Invalid token')
    }

    const meData = await meRes.json()

    const newUser: User = {
      id: meData.id,
      full_name: meData.full_name,
      email: meData.email,
      is_active: meData.is_active,
      is_admin: meData.is_admin,
      access_token: access_token
    }

    setUser(newUser)
    localStorage.setItem('user', JSON.stringify(newUser))
  }

  // Função de login
  async function login(email: string, password: string) {
    const loginRes = await fetch(`${environments.apiUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    })

    if (!loginRes.ok) {
      const error = await loginRes.json()
      throw new Error(error.detail || 'Login failed')
    }

    const tokenData: Token = await loginRes.json()
    await me(tokenData.access_token)
  }

  // Função de registro
  async function register(full_name: string, email: string, password: string) {
    const registerRes = await fetch(`${environments.apiUrl}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_name, email, password })
    })

    if (!registerRes.ok) {
      const error = await registerRes.json()
      throw new Error(error.detail || 'Registration failed')
    }

    // Após registrar, faz login automaticamente
    await login(email, password)
  }

  // Logout
  function logout() {
    setUser(null)
    localStorage.removeItem('user')
    navigate('/login')
  }

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}
