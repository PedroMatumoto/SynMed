import { createContext, useContext, useState, ReactNode } from 'react'
import {
  DrugEffectRequest,
  DrugEffectResponse,
  DrugSearchResult,
  EffectAnalysisHistory
} from '../types/analysis'
import { environments } from '@/utils/env/environments'
import { useAuth } from './authContext'
import { useToast } from '@/utils/useToast'

interface AnalysisContextType {
  currentAnalysis: DrugEffectResponse | null
  analysisHistory: EffectAnalysisHistory[]
  loading: boolean
  error: string | null
  toasts: ReturnType<typeof useToast>
  searchDrugs: (query: string, useSemanticSearch?: boolean) => Promise<DrugSearchResult[]>
  analyzeEffect: (request: DrugEffectRequest) => Promise<void>
  fetchHistory: () => Promise<void>
  clearHistory: () => Promise<void>
  deleteHistoryItem: (id: string) => Promise<void>
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined)

export function useAnalysis() {
  const context = useContext(AnalysisContext)
  if (!context) {
    throw new Error('useAnalysis must be used within AnalysisProvider')
  }
  return context
}

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [currentAnalysis, setCurrentAnalysis] = useState<DrugEffectResponse | null>(null)
  const [analysisHistory, setAnalysisHistory] = useState<EffectAnalysisHistory[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { user } = useAuth()
  const toasts = useToast()

  async function searchDrugs(
    query: string,
    useSemanticSearch: boolean = true
  ): Promise<DrugSearchResult[]> {
    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }

      if (user?.access_token) {
        headers['Authorization'] = `Bearer ${user.access_token}`
      }

      const response = await fetch(
        `${environments.apiUrl}/drugs/search?query=${encodeURIComponent(query)}&use_semantic=${useSemanticSearch}`,
        { headers }
      )

      if (!response.ok) {
        throw new Error('Erro ao buscar medicamentos')
      }

      return await response.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido')
      return []
    }
  }

  async function analyzeEffect(request: DrugEffectRequest): Promise<void> {
    setLoading(true)
    setError(null)

    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }

      if (user?.access_token) {
        headers['Authorization'] = `Bearer ${user.access_token}`
      }

      const response = await fetch(`${environments.apiUrl}/analyze-effect`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          ...request,
          use_semantic_search: request.use_semantic_search ?? true,
          similarity_threshold: request.similarity_threshold ?? 0.6,
          use_gemma_analysis: request.use_gemma_analysis ?? true
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Erro ao analisar efeito')
      }

      const data: DrugEffectResponse = await response.json()
      setCurrentAnalysis(data)

      // Mostra toast de sucesso
      toasts.showSuccess(
        'Análise concluída com sucesso!',
        `A análise do medicamento "${data.drug_name}" foi realizada.`,
        6000
      )

      // Atualiza o histórico se o usuário estiver logado
      if (user) {
        await fetchHistory()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido')
      throw err
    } finally {
      setLoading(false)
    }
  }

  async function fetchHistory(): Promise<void> {
    if (!user?.access_token) return

    try {
      const response = await fetch(`${environments.apiUrl}/my-analyses?limit=50`, {
        headers: {
          Authorization: `Bearer ${user.access_token}`
        }
      })

      if (!response.ok) {
        throw new Error('Erro ao buscar histórico')
      }

      const data = await response.json()
      setAnalysisHistory(data)
    } catch (err) {
      console.error('Error fetching history:', err)
    }
  }

  async function clearHistory(): Promise<void> {
    if (!user?.access_token) return

    try {
      const response = await fetch(`${environments.apiUrl}/my-analyses`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${user.access_token}`
        }
      })

      if (!response.ok) {
        throw new Error('Erro ao limpar histórico')
      }

      setAnalysisHistory([])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido')
      throw err
    }
  }

  async function deleteHistoryItem(id: string): Promise<void> {
    if (!user?.access_token) return

    try {
      const response = await fetch(`${environments.apiUrl}/my-analyses/${id}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${user.access_token}`
        }
      })

      if (!response.ok) {
        throw new Error('Erro ao deletar item do histórico')
      }

      setAnalysisHistory((prev) => prev.filter((item) => item.id !== id))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido')
      throw err
    }
  }

  return (
    <AnalysisContext.Provider
      value={{
        currentAnalysis,
        analysisHistory,
        loading,
        error,
        toasts,
        searchDrugs,
        analyzeEffect,
        fetchHistory,
        clearHistory,
        deleteHistoryItem
      }}
    >
      {children}
    </AnalysisContext.Provider>
  )
}
