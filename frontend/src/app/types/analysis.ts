export interface DrugEffectRequest {
  drug_name: string
  effect_symptom: string  // Mantido para compatibilidade
  symptoms?: string[]  // Lista de sintomas múltiplos
  natural_language_text?: string  // Texto em linguagem natural
  use_semantic_search?: boolean
  similarity_threshold?: number
  use_gemma_analysis?: boolean
}

export interface SimilarEffect {
  effect: string
  similarity_score: number
  effect_pt?: string
}

export interface SymptomAnalysis {
  original_symptom: string
  translated_symptom?: string
  similar_effects: SimilarEffect[]
  confidence: string
  similarity_score: number
}

export interface ExtractedSymptom {
  text: string
  confidence_score: number
  source_span: string
}

export interface DrugInfo {
  drug_class?: string
  indications?: string
  side_effect_severity?: string
}

export interface DrugEffectResponse {
  drug_name: string
  user_effect: string  // Mantido para compatibilidade
  translated_effect?: string
  possible_translations?: Array<{ [key: string]: number }>
  similar_effects: SimilarEffect[]  // Mantido para compatibilidade
  confidence: string  // Confiança geral
  similarity_score: number  // Score geral
  drug_info: DrugInfo
  basic_analysis: string
  gemma_analysis?: string
  all_known_effects?: string[]
  
  // Novos campos para múltiplos sintomas
  extracted_symptoms?: ExtractedSymptom[]
  symptom_analyses?: SymptomAnalysis[]
  overall_confidence?: string  // Confiança considerando todos os sintomas
}

export interface DrugSearchResult {
  drug_name: string
  similarity_score?: number
  source: string
}

export interface EffectAnalysisHistory {
  id?: string
  user_id: string
  drug_name: string
  effect_symptom: string  // Mantido para compatibilidade
  symptoms?: string[]  // Lista de sintomas analisados
  extracted_symptoms?: ExtractedSymptom[]  // Sintomas extraídos
  natural_language_input?: string  // Texto original em linguagem natural
  confidence: string
  similarity_score: number
  overall_confidence?: string
  use_semantic_search: boolean
  use_gemma_analysis: boolean
  timestamp: string
}
