export interface DrugEffectRequest {
  drug_name: string
  effect_symptom: string
  use_semantic_search?: boolean
  similarity_threshold?: number
  use_gemma_analysis?: boolean
}

export interface SimilarEffect {
  effect: string
  similarity_score: number
  effect_pt?: string
}

export interface DrugInfo {
  drug_class?: string
  indications?: string
  side_effect_severity?: string
}

export interface DrugEffectResponse {
  drug_name: string
  user_effect: string
  translated_effect?: string
  possible_translations?: Array<{ [key: string]: number }>
  similar_effects: SimilarEffect[]
  confidence: string
  similarity_score: number
  drug_info: DrugInfo
  basic_analysis: string
  gemma_analysis?: string
  all_known_effects?: string[]
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
  effect_symptom: string
  confidence: string
  similarity_score: number
  use_semantic_search: boolean
  use_gemma_analysis: boolean
  timestamp: string
}
