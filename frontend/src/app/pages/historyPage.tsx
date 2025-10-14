import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/authContext'
import { useAnalysis } from '../contexts/analysisContext'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faPills,
  faTrash,
  faSearch,
  faSignOutAlt,
  faHistory,
  faCalendarAlt,
  faChartBar
} from '@fortawesome/free-solid-svg-icons'

export default function HistoryPage() {
  const { user, logout } = useAuth()
  const { analysisHistory, loading, fetchHistory, clearHistory, deleteHistoryItem } =
    useAnalysis()
  const navigate = useNavigate()

  useEffect(() => {
    fetchHistory()
  }, [])

  async function handleClearHistory() {
    if (window.confirm('Tem certeza que deseja limpar todo o histórico?')) {
      try {
        await clearHistory()
      } catch (err) {
        alert('Erro ao limpar histórico')
      }
    }
  }

  async function handleDeleteItem(id: string) {
    if (window.confirm('Tem certeza que deseja deletar este item?')) {
      try {
        await deleteHistoryItem(id)
      } catch (err) {
        alert('Erro ao deletar item')
      }
    }
  }

  function formatDate(dateString: string) {
    const date = new Date(dateString)
    return date.toLocaleString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="flex min-h-screen w-screen flex-col bg-gradient-to-br from-dark-100 to-white font-sans">
      {/* Header */}
      <div className="flex h-20 w-full items-center justify-between border-b border-dark-200 bg-white/95 backdrop-blur-sm px-8 shadow-lg">
        <div className="flex items-center gap-4">
          <div className="rounded-xl bg-accent/10 p-2">
            <FontAwesomeIcon icon={faPills} className="text-2xl text-accent" />
          </div>
          <h1 className="text-2xl font-bold text-dark-800">SynMed</h1>
        </div>

        <div className="flex items-center gap-6">
          <div className="group flex items-center gap-3 rounded-lg bg-dark-100 px-4 py-2 text-dark-700">
            <div className="inline-block origin-bottom group-hover:animate-wave">👋</div>
            <span className="font-medium">{user?.full_name}</span>
          </div>

          <button
            onClick={() => navigate('/analysis')}
            className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-white transition-all hover:bg-accent/90 hover:scale-105"
          >
            <FontAwesomeIcon icon={faSearch} />
            Nova Análise
          </button>

          <button
            onClick={logout}
            className="flex items-center gap-2 rounded-lg border border-dark-300 bg-white px-4 py-2 text-dark-700 transition-all hover:bg-dark-100"
          >
            <FontAwesomeIcon icon={faSignOutAlt} />
            Sair
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 flex-col items-center justify-start p-8">
        <div className="w-full max-w-6xl">
          <div className="mb-8">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-dark-800/10 p-3">
                  <FontAwesomeIcon icon={faCalendarAlt} className="text-2xl text-dark-800" />
                </div>
                <div>
                  <h2 className="text-4xl font-bold text-dark-800">Histórico de Análises</h2>
                  <p className="text-dark-600 mt-1">Acompanhe suas consultas anteriores</p>
                </div>
              </div>

              {analysisHistory.length > 0 && (
                <button
                  onClick={handleClearHistory}
                  className="flex items-center gap-2 rounded-lg bg-red-500 px-4 py-2 text-white transition-all hover:bg-red-600 hover:scale-105"
                >
                  <FontAwesomeIcon icon={faTrash} />
                  Limpar Histórico
                </button>
              )}
            </div>

            {/* Stats Cards */}
            {analysisHistory.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="rounded-xl bg-white/80 backdrop-blur-sm border border-dark-200 p-6 shadow-lg">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-blue-500/10 p-2">
                      <FontAwesomeIcon icon={faChartBar} className="text-blue-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-dark-800">{analysisHistory.length}</p>
                      <p className="text-dark-600 text-sm">Análises realizadas</p>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl bg-white/80 backdrop-blur-sm border border-dark-200 p-6 shadow-lg">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-green-500/10 p-2">
                      <FontAwesomeIcon icon={faPills} className="text-green-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-dark-800">
                        {new Set(analysisHistory.map(item => item.drug_name)).size}
                      </p>
                      <p className="text-dark-600 text-sm">Medicamentos únicos</p>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl bg-white/80 backdrop-blur-sm border border-dark-200 p-6 shadow-lg">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-purple-500/10 p-2">
                      <FontAwesomeIcon icon={faHistory} className="text-purple-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-dark-800">
                        {analysisHistory.filter(item => item.confidence === 'Alta').length}
                      </p>
                      <p className="text-dark-600 text-sm">Alta confiança</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {loading ? (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="h-12 w-12 animate-spin rounded-full border-4 border-accent/20 border-t-accent mb-4"></div>
              <p className="text-dark-600">Carregando histórico...</p>
            </div>
          ) : analysisHistory.length === 0 ? (
            <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-dark-200 p-16 text-center shadow-lg">
              <div className="mb-6">
                <div className="rounded-full bg-dark-100 p-6 mx-auto w-fit mb-4">
                  <FontAwesomeIcon icon={faSearch} className="text-4xl text-dark-400" />
                </div>
                <h3 className="text-2xl font-bold text-dark-800 mb-2">Nenhuma análise encontrada</h3>
                <p className="text-dark-600 mb-6">Comece fazendo sua primeira análise de medicamentos</p>
              </div>
              <button
                onClick={() => navigate('/analysis')}
                className="rounded-xl bg-accent px-8 py-4 text-white font-semibold transition-all hover:bg-accent/90 hover:scale-105"
              >
                Fazer primeira análise
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {analysisHistory.map((item) => (
                <div
                  key={item.id}
                  className="rounded-2xl bg-white/80 backdrop-blur-sm border border-dark-200 p-6 shadow-lg transition-all hover:shadow-xl hover:scale-[1.01]"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="mb-4 flex items-center gap-4">
                        <div className="rounded-lg bg-accent/10 p-2">
                          <FontAwesomeIcon icon={faPills} className="text-accent" />
                        </div>
                        <div>
                          <h3 className="text-2xl font-bold text-dark-800">
                            {item.drug_name}
                          </h3>
                          <span
                            className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-bold border ${
                              item.confidence === 'Alta'
                                ? 'bg-green-50 text-green-700 border-green-200'
                                : item.confidence === 'Moderada'
                                  ? 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                  : 'bg-red-50 text-red-700 border-red-200'
                            }`}
                          >
                            Confiança: {item.confidence}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="rounded-lg bg-dark-50 p-4">
                          <p className="font-semibold text-dark-800 mb-1">Sintoma Relatado:</p>
                          <p className="text-dark-700">{item.effect_symptom}</p>
                        </div>

                        <div className="rounded-lg bg-dark-50 p-4">
                          <p className="font-semibold text-dark-800 mb-1">Score de Similaridade:</p>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-2 bg-dark-200 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-accent to-accent/80 rounded-full transition-all duration-1000"
                                style={{ width: `${item.similarity_score * 100}%` }}
                              />
                            </div>
                            <span className="font-bold text-accent">
                              {(item.similarity_score * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-3 text-sm">
                        <span className={`flex items-center gap-1 px-3 py-1 rounded-full ${
                          item.use_semantic_search 
                            ? 'bg-blue-100 text-blue-700' 
                            : 'bg-gray-100 text-gray-600'
                        }`}>
                          {item.use_semantic_search ? '✓' : '✗'} Busca semântica
                        </span>
                        <span className={`flex items-center gap-1 px-3 py-1 rounded-full ${
                          item.use_gemma_analysis 
                            ? 'bg-purple-100 text-purple-700' 
                            : 'bg-gray-100 text-gray-600'
                        }`}>
                          {item.use_gemma_analysis ? '✓' : '✗'} Análise IA
                        </span>
                        <span className="flex items-center gap-1 px-3 py-1 rounded-full bg-gray-100 text-gray-600">
                          <FontAwesomeIcon icon={faCalendarAlt} className="text-xs" />
                          {formatDate(item.timestamp)}
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={() => handleDeleteItem(item.id!)}
                      className="ml-4 rounded-lg p-3 text-red-500 transition-all hover:bg-red-50 hover:scale-110"
                    >
                      <FontAwesomeIcon icon={faTrash} className="text-lg" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
