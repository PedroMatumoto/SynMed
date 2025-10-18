import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/authContext'
import { useAnalysis } from '../contexts/analysisContext'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faPills,
  faSearch,
  faHistory,
  faSignOutAlt
} from '@fortawesome/free-solid-svg-icons'
import { DrugEffectRequest } from '../types/analysis'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ToastContainer } from '../components/ui/toastContainer'

export default function AnalysisPage() {
  const { user, logout } = useAuth()
  const { currentAnalysis, loading, error, toasts, searchDrugs, analyzeEffect } = useAnalysis()
  const navigate = useNavigate()

  const [drugName, setDrugName] = useState('')
  const [effectSymptom, setEffectSymptom] = useState('')
  const [useSemanticSearch, setUseSemanticSearch] = useState(true)
  const [useGemmaAnalysis, setUseGemmaAnalysis] = useState(true)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [suggestionSelected, setSuggestionSelected] = useState(false)

  useEffect(() => {
    async function fetchSuggestions() {
      // Não busca sugestões se uma sugestão foi selecionada recentemente
      if (suggestionSelected) {
        return
      }
      
      if (drugName.length >= 2) {
        const results = await searchDrugs(drugName, useSemanticSearch)
        setSuggestions(results.map((r) => r.drug_name).slice(0, 5))
        setShowSuggestions(true)
      } else {
        setSuggestions([])
        setShowSuggestions(false)
      }
    }

    const timeoutId = setTimeout(fetchSuggestions, 300)
    return () => clearTimeout(timeoutId)
  }, [drugName, useSemanticSearch, searchDrugs, suggestionSelected])

  async function handleAnalysis() {
    if (!drugName || !effectSymptom) {
      toasts.showWarning(
        'Campos obrigatórios',
        'Por favor, preencha o nome do medicamento e o sintoma'
      )
      return
    }

    const request: DrugEffectRequest = {
      drug_name: drugName,
      effect_symptom: effectSymptom,
      use_semantic_search: useSemanticSearch,
      use_gemma_analysis: useGemmaAnalysis
    }

    try {
      await analyzeEffect(request)
    } catch (err) {
      console.error('Error analyzing effect:', err)
      toasts.showError(
        'Erro na análise',
        'Ocorreu um erro ao processar sua solicitação. Tente novamente.'
      )
    }
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
            onClick={() => navigate('/history')}
            className="flex items-center gap-2 rounded-lg bg-dark-800 px-4 py-2 text-white transition-all hover:bg-dark-700 hover:scale-105"
          >
            <FontAwesomeIcon icon={faHistory} />
            Histórico
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
        <div className="w-full max-w-4xl">
          <div className="mb-8 text-center">
            <h2 className="text-4xl font-bold text-dark-800 mb-4">
              Análise de Efeitos Colaterais
            </h2>
            <p className="text-lg text-dark-600 max-w-2xl mx-auto">
              Verifique se seus sintomas podem estar relacionados a um medicamento usando nossa IA avançada
            </p>
          </div>

          {/* Analysis Form */}
          <div className="mb-8 rounded-2xl bg-white/80 backdrop-blur-sm border border-dark-200 p-8 shadow-2xl">
            <div className="mb-6">
              <label className="mb-3 block text-lg font-semibold text-dark-800">
                💊 Nome do Medicamento
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={drugName}
                  onChange={(e) => {
                    setDrugName(e.target.value)
                    setSuggestionSelected(false) // Reset quando usuário digita
                  }}
                  onFocus={() => {
                    setSuggestionSelected(false) // Reset quando usuário foca no campo
                    setShowSuggestions(suggestions.length > 0)
                  }}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                  className="w-full rounded-xl border-2 border-dark-200 bg-white px-4 py-4 text-lg outline-none transition-all focus:border-accent focus:shadow-lg focus:bg-white"
                  placeholder="Ex: Amoxicilina, Losartana..."
                />

                {showSuggestions && suggestions.length > 0 && (
                  <div className="absolute z-10 mt-2 w-full rounded-xl border border-dark-300 bg-white shadow-2xl overflow-hidden">
                    {suggestions.map((suggestion, index) => (
                      <div
                        key={index}
                        onClick={() => {
                          setDrugName(suggestion)
                          setShowSuggestions(false)
                          setSuggestionSelected(true) // Marca que uma sugestão foi selecionada
                        }}
                        className="cursor-pointer px-4 py-3 hover:bg-accent/10 transition-colors border-b border-dark-100 last:border-b-0"
                      >
                        {suggestion}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mb-6">
              <label className="mb-3 block text-lg font-semibold text-dark-800">
                🔍 Sintoma ou Efeito Observado
              </label>
              <textarea
                value={effectSymptom}
                onChange={(e) => setEffectSymptom(e.target.value)}
                className="w-full rounded-xl border-2 border-dark-200 bg-white px-4 py-4 text-lg outline-none transition-all focus:border-accent focus:shadow-lg focus:bg-white resize-none"
                placeholder="Descreva detalhadamente os sintomas que você está sentindo (ex: dor de cabeça intensa, náusea após as refeições, tontura ao levantar...)"
                rows={4}
              />
            </div>

            <div className="mb-8">
              <h3 className="mb-4 text-lg font-semibold text-dark-800">⚙️ Opções de Análise</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="flex items-center gap-3 rounded-lg bg-dark-50 p-4 border border-dark-200 cursor-pointer hover:bg-dark-100 transition-colors">
                  <input
                    type="checkbox"
                    checked={useSemanticSearch}
                    onChange={(e) => setUseSemanticSearch(e.target.checked)}
                    className="h-5 w-5 rounded border-2 border-dark-300 text-accent focus:ring-accent focus:ring-2"
                  />
                  <div>
                    <span className="font-medium text-dark-800">Busca semântica</span>
                    <p className="text-sm text-dark-600">Análise mais precisa usando similaridade de contexto</p>
                  </div>
                </label>

                <label className="flex items-center gap-3 rounded-lg bg-dark-50 p-4 border border-dark-200 cursor-pointer hover:bg-dark-100 transition-colors">
                  <input
                    type="checkbox"
                    checked={useGemmaAnalysis}
                    onChange={(e) => setUseGemmaAnalysis(e.target.checked)}
                    className="h-5 w-5 rounded border-2 border-dark-300 text-accent focus:ring-accent focus:ring-2"
                  />
                  <div>
                    <span className="font-medium text-dark-800">Análise com IA (Gemma)</span>
                    <p className="text-sm text-dark-600">Interpretação avançada usando inteligência artificial</p>
                  </div>
                </label>
              </div>
            </div>

            <button
              onClick={handleAnalysis}
              disabled={loading}
              className="group flex w-full items-center justify-center gap-4 rounded-xl bg-gradient-to-r from-accent to-accent/80 py-5 text-xl font-bold text-white transition-all duration-300 hover:from-accent/90 hover:to-accent/70 hover:scale-[1.02] hover:shadow-2xl disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
            >
              {loading ? (
                <>
                  <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  Analisando dados...
                </>
              ) : (
                <>
                  <FontAwesomeIcon icon={faSearch} className="text-2xl" />
                  Iniciar Análise
                </>
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-8 rounded-xl bg-red-500/10 border border-red-500/20 p-6 text-center">
              <div className="text-red-600 font-medium text-lg">⚠️ Erro na análise</div>
              <div className="text-red-500 mt-2">{error}</div>
            </div>
          )}

          {/* Results */}
          {currentAnalysis && (
            <div className="animate-fadeIn rounded-2xl bg-white/90 backdrop-blur-sm border border-dark-200 p-8 shadow-2xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="rounded-xl bg-green-500/10 p-3">
                  <FontAwesomeIcon icon={faPills} className="text-2xl text-green-600" />
                </div>
                <h3 className="text-3xl font-bold text-dark-800">
                  Resultado da Análise
                </h3>
              </div>

              <div className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="rounded-xl bg-dark-50 p-6 border border-dark-200">
                  <h4 className="font-semibold text-dark-800 mb-3 flex items-center gap-2">
                    💊 Medicamento Analisado
                  </h4>
                  <p className="text-xl font-bold text-accent">{currentAnalysis.drug_name}</p>
                </div>

                <div className="rounded-xl bg-dark-50 p-6 border border-dark-200">
                  <h4 className="font-semibold text-dark-800 mb-3 flex items-center gap-2">
                    📊 Nível de Confiança
                  </h4>
                  <span
                    className={`inline-flex items-center rounded-full px-4 py-2 text-lg font-bold ${
                      currentAnalysis.confidence === 'Alta'
                        ? 'bg-green-100 text-green-700 border border-green-200'
                        : currentAnalysis.confidence === 'Moderada'
                          ? 'bg-yellow-100 text-yellow-700 border border-yellow-200'
                          : 'bg-red-100 text-red-700 border border-red-200'
                    }`}
                  >
                    {currentAnalysis.confidence}
                  </span>
                </div>

                <div className="rounded-xl bg-dark-50 p-6 border border-dark-200">
                  <h4 className="font-semibold text-dark-800 mb-3 flex items-center gap-2">
                    🎯 Sintoma Relatado
                  </h4>
                  <p className="text-dark-700">{currentAnalysis.user_effect}</p>
                </div>

                <div className="rounded-xl bg-dark-50 p-6 border border-dark-200">
                  <h4 className="font-semibold text-dark-800 mb-3 flex items-center gap-2">
                    🔢 Score de Similaridade
                  </h4>
                  <p className="text-2xl font-bold text-accent">
                    {(currentAnalysis.similarity_score * 100).toFixed(1)}%
                  </p>
                </div>

                {currentAnalysis.translated_effect && (
                  <div className="md:col-span-2 rounded-xl bg-blue-50 p-6 border border-blue-200">
                    <h4 className="font-semibold text-dark-800 mb-3 flex items-center gap-2">
                      🌐 Tradução (EN)
                    </h4>
                    <p className="text-dark-700 italic">{currentAnalysis.translated_effect}</p>
                  </div>
                )}
              </div>

              {currentAnalysis.similar_effects.length > 0 && (
                <div className="mb-8">
                  <h4 className="mb-4 text-2xl font-bold text-dark-800 flex items-center gap-2">
                    🔍 Efeitos Similares Encontrados
                  </h4>
                  <div className="space-y-3">
                    {currentAnalysis.similar_effects.slice(0, 5).map((effect, index) => (
                      <div key={index} className="flex items-center justify-between rounded-xl bg-gradient-to-r from-accent/5 to-accent/10 border border-accent/20 p-4 hover:from-accent/10 hover:to-accent/15 transition-all">
                        <div className="flex flex-col">
                          <span className="font-medium text-dark-800">
                            {effect.effect_pt || effect.effect}
                          </span>
                          {effect.effect_pt && (
                            <span className="text-sm text-dark-500 italic">
                              Original: {effect.effect}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-dark-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-accent to-accent/80 rounded-full transition-all duration-1000"
                              style={{ width: `${effect.similarity_score * 100}%` }}
                            />
                          </div>
                          <span className="font-bold text-accent min-w-[4rem] text-right">
                            {(effect.similarity_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {currentAnalysis.drug_info && (
                <div className="mb-8">
                  <h4 className="mb-4 text-2xl font-bold text-dark-800 flex items-center gap-2">
                    ℹ️ Informações do Medicamento
                  </h4>
                  <div className="rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {currentAnalysis.drug_info.drug_class && (
                        <div>
                          <span className="font-semibold text-dark-800">Classe:</span>
                          <p className="text-dark-700 mt-1">{currentAnalysis.drug_info.drug_class}</p>
                        </div>
                      )}
                      {currentAnalysis.drug_info.side_effect_severity && (
                        <div>
                          <span className="font-semibold text-dark-800">Severidade:</span>
                          <p className="text-dark-700 mt-1">{currentAnalysis.drug_info.side_effect_severity}</p>
                        </div>
                      )}
                    </div>
                    {currentAnalysis.drug_info.indications && (
                      <div className="mt-4">
                        <span className="font-semibold text-dark-800">Indicações:</span>
                        <p className="text-dark-700 mt-1">{currentAnalysis.drug_info.indications}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="mb-8">
                <h4 className="mb-4 text-2xl font-bold text-dark-800 flex items-center gap-2">
                  📋 Análise Básica
                </h4>
                <div className="rounded-xl bg-gradient-to-br from-green-50 to-green-100 border border-green-200 p-6">
                  <div className="prose prose-sm max-w-none text-dark-700 leading-relaxed">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {currentAnalysis.basic_analysis}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>

              {currentAnalysis.gemma_analysis && (
                <div className="mb-8">
                  <h4 className="mb-4 text-2xl font-bold text-dark-800 flex items-center gap-2">
                    🤖 Análise com IA (Gemma)
                  </h4>
                  <div className="rounded-xl bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 p-6">
                    <div className="prose prose-sm max-w-none text-dark-700 leading-relaxed">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {currentAnalysis.gemma_analysis}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              )}

              {/* Important Notice */}
              <div className="rounded-xl bg-yellow-50 border border-yellow-200 p-6">
                <div className="flex items-start gap-3">
                  <div className="text-2xl">⚠️</div>
                  <div>
                    <h5 className="font-bold text-yellow-800 mb-2">Importante - Aviso Médico</h5>
                    <p className="text-yellow-700 text-sm leading-relaxed">
                      Esta análise é apenas informativa e não substitui a consulta médica. 
                      Se você está experienciando efeitos adversos de medicamentos, consulte 
                      imediatamente um profissional de saúde qualificado. Não interrompa 
                      medicamentos prescritos sem orientação médica.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Toast Container */}
      <ToastContainer 
        toasts={toasts.toasts} 
        onClose={toasts.removeToast} 
      />
    </div>
  )
}
