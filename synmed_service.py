import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict, Optional
import re
import pickle
import os
import hashlib
from api_models import DrugInfo, SimilarEffect, DrugSearchResult
from app.gemma_med import GemmaMedClient


class SynMedService:
    """Serviço principal para análise de efeitos colaterais de medicamentos"""
    
    def __init__(self):
        self._sentence_model = None
        self._drug_model = None
        self._drug_names_df = None
        self._realistic_drugs_df = None
        self._sider_data_df = None
        self._pt_to_en_dict = {}
        self._en_to_pt_dict = {}
        self._gemma_client = None
        self._initialize_models_and_data()
    
    def _initialize_models_and_data(self):
        """Inicializa modelos e carrega dados"""
        try:
            self._sentence_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
            self._drug_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
            
            self._load_drug_data()
            self._load_translations()
            
            try:
                import os
                if os.path.exists('app/gemma_med.py'):
                    from app.gemma_med import get_gemma_client
                    self._gemma_client = get_gemma_client()
                elif os.path.exists('gemma_med.py'):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("gemma_med", "gemma_med.py")
                    gemma_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(gemma_mod)
                    self._gemma_client = gemma_mod.get_gemma_client()
                else:
                    self._gemma_client = None
            except Exception as e:
                print(f"Aviso: Não foi possível carregar GemmaMed: {e}")
                self._gemma_client = None
                
        except Exception as e:
            print(f"Erro ao inicializar SynMedService: {e}")
            raise
    
    def _load_drug_data(self):
        """Carrega dados de medicamentos"""
        try:
            self._drug_names_df = pd.read_csv('data/drug_names.csv', sep=';', header=None, names=['CID', 'drug_name'])
            
            realistic_drugs = pd.read_csv('data/realistic_drug_labels_side_effects.csv')
            self._realistic_drugs_df = self._treat_realistic_drugs(realistic_drugs)
            
            self._sider_data_df = pd.read_csv('data/meddra_all_se.csv', sep=';', header=None, 
                                            names=['CID', 'STITCH_ID', 'UMLS_ID', 'MedDRA_type', 'UMLS_ID2', 'side_effect'])
        except Exception as e:
            print(f"Erro ao carregar dados de medicamentos: {e}")
            raise
    
    def _load_translations(self):
        """Carrega traduções de efeitos colaterais"""
        try:
            translations_df = pd.read_csv('data/side_effects_translated.csv')
            self._pt_to_en_dict = dict(zip(translations_df['Traduzido'].str.lower(), translations_df['Original']))
            self._en_to_pt_dict = dict(zip(translations_df['Original'].str.lower(), translations_df['Traduzido']))
        except Exception as e:
            print(f"Aviso: Não foi possível carregar traduções: {e}")
            self._pt_to_en_dict = {}
            self._en_to_pt_dict = {}
    
    def _treat_realistic_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata dados de medicamentos realistas"""
        df['side_effects'] = df['side_effects'].fillna('').apply(lambda x: ', '.join([se.strip() for se in str(x).split(',') if se.strip() != '']))
        df = df[df['side_effects'] != '']
        df['drug_name'] = df['drug_name'].str.replace(r'\d+', '', regex=True).str.strip()
        df['side_effects'] = df['side_effects'].str.split(', ')
        df = df.explode('side_effects')
        df = df.reset_index(drop=True)
        return df
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para busca"""
        return re.sub(r'[^\w\s]', '', text.lower().strip())
    
    def _get_cache_key(self, drug_names: List[str]) -> str:
        """Gera chave de cache para embeddings"""
        combined_names = ''.join(sorted(drug_names))
        return hashlib.md5(combined_names.encode()).hexdigest()
    
    def _load_drug_embeddings_cache(self, cache_key: str) -> Optional[Dict]:
        """Carrega embeddings do cache"""
        cache_file = f"cache/drug_embeddings_{cache_key}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_drug_embeddings_cache(self, cache_key: str, embeddings_data: Dict):
        """Salva embeddings no cache"""
        os.makedirs("cache", exist_ok=True)
        cache_file = f"cache/drug_embeddings_{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
    
    def _create_drug_embeddings(self, drug_names: List[str]) -> Tuple[np.ndarray, Dict]:
        """Cria embeddings para nomes de medicamentos"""
        cache_key = self._get_cache_key(drug_names)
        
        cached_data = self._load_drug_embeddings_cache(cache_key)
        if cached_data is not None:
            return cached_data['embeddings'], cached_data['name_to_index']
        
        clean_names = [name.strip().lower() for name in drug_names if name and str(name) != 'nan']
        embeddings = self._drug_model.encode(clean_names, convert_to_tensor=False)
        
        name_to_index = {name: idx for idx, name in enumerate(clean_names)}
        
        embeddings_data = {
            'embeddings': embeddings,
            'name_to_index': name_to_index,
            'drug_names': clean_names
        }
        self._save_drug_embeddings_cache(cache_key, embeddings_data)
        
        return embeddings, name_to_index
    
    def find_drug_matches(self, drug_input: str, use_semantic: bool = True, threshold: float = 0.6) -> List[DrugSearchResult]:
        """Encontra medicamentos correspondentes"""
        if use_semantic and self._drug_model is not None:
            return self._semantic_drug_search(drug_input, threshold)
        else:
            return self._exact_drug_search(drug_input)
    
    def _exact_drug_search(self, drug_input: str) -> List[DrugSearchResult]:
        """Busca exata por medicamentos"""
        normalized_input = self._normalize_text(drug_input)
        matches = []
        
        for _, row in self._drug_names_df.iterrows():
            if normalized_input in self._normalize_text(row['drug_name']):
                matches.append(DrugSearchResult(
                    drug_name=row['drug_name'],
                    source="official"
                ))
        
        for _, row in self._realistic_drugs_df.iterrows():
            if normalized_input in self._normalize_text(row['drug_name']):
                matches.append(DrugSearchResult(
                    drug_name=row['drug_name'],
                    source="realistic"
                ))
        
        seen = set()
        unique_matches = []
        for match in matches:
            if match.drug_name not in seen:
                seen.add(match.drug_name)
                unique_matches.append(match)
        
        return unique_matches
    
    def _semantic_drug_search(self, query: str, threshold: float = 0.6, top_k: int = 10) -> List[DrugSearchResult]:
        """Busca semântica por medicamentos"""
        all_drugs = []
        drug_sources = {}
        
        for _, row in self._drug_names_df.iterrows():
            drug_name = str(row['drug_name']).strip()
            if drug_name and drug_name != 'nan':
                all_drugs.append(drug_name)
                drug_sources[drug_name] = 'official'
        
        for _, row in self._realistic_drugs_df.iterrows():
            drug_name = str(row['drug_name']).strip()
            if drug_name and drug_name != 'nan' and drug_name not in drug_sources:
                all_drugs.append(drug_name)
                drug_sources[drug_name] = 'realistic'
        
        if not all_drugs:
            return []
        
        drug_embeddings, name_to_index = self._create_drug_embeddings(all_drugs)
        
        query_embedding = self._drug_model.encode([query.strip().lower()], convert_to_tensor=False)
        
        similarities = util.cos_sim(query_embedding, drug_embeddings)[0]
        
        results = []
        for drug_name, idx in name_to_index.items():
            similarity = similarities[idx].item()
            if similarity >= threshold:
                original_name = drug_name
                for original_drug in all_drugs:
                    if original_drug.lower() == drug_name:
                        original_name = original_drug
                        break
                
                results.append(DrugSearchResult(
                    drug_name=original_name,
                    similarity_score=similarity,
                    source=drug_sources.get(original_name, 'unknown')
                ))
        
        results.sort(key=lambda x: x.similarity_score or 0, reverse=True)
        return results[:top_k]
    
    def get_side_effects_for_drug(self, drug_name: str) -> List[str]:
        """Obtém efeitos colaterais conhecidos para um medicamento"""
        side_effects = []
        
        realistic_match = self._realistic_drugs_df[
            self._realistic_drugs_df['drug_name'].str.contains(drug_name, case=False, na=False)
        ]
        if not realistic_match.empty:
            for _, row in realistic_match.iterrows():
                if str(row['side_effects']) != 'nan':
                    side_effects.append(str(row['side_effects']).strip())
        
        drug_cid = self._drug_names_df[
            self._drug_names_df['drug_name'].str.contains(drug_name, case=False, na=False)
        ]
        if not drug_cid.empty:
            cid = drug_cid.iloc[0]['CID']
            sider_effects = self._sider_data_df[self._sider_data_df['CID'] == cid]['side_effect'].tolist()
            side_effects.extend(sider_effects)
        
        return list(set([effect for effect in side_effects if effect and str(effect) != 'nan']))
    
    def get_drug_info(self, drug_name: str) -> DrugInfo:
        """Obtém informações do medicamento"""
        drug_row = self._realistic_drugs_df[
            self._realistic_drugs_df['drug_name'].str.contains(drug_name, case=False, na=False)
        ]
        
        if not drug_row.empty:
            row_data = drug_row.iloc[0].to_dict()
            return DrugInfo(
                drug_class=row_data.get('drug_class'),
                indications=row_data.get('indications'),
                side_effect_severity=row_data.get('side_effect_severity')
            )
        
        return DrugInfo()
    
    def translate_symptom_to_english(self, symptom: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Traduz sintoma em português para inglês"""
        normalized_symptom = self._normalize_text(symptom)
        
        if normalized_symptom in self._pt_to_en_dict:
            return self._pt_to_en_dict[normalized_symptom], [(self._pt_to_en_dict[normalized_symptom], 1.0)]
        
        partial_matches = []
        for pt_effect, en_effect in self._pt_to_en_dict.items():
            if normalized_symptom in pt_effect or pt_effect in normalized_symptom:
                partial_matches.append((en_effect, 0.9))
        
        if partial_matches:
            return partial_matches[0][0], partial_matches
        
        if self._pt_to_en_dict and self._sentence_model:
            symptom_embedding = self._sentence_model.encode(normalized_symptom, convert_to_tensor=True)
            semantic_matches = []
            
            for pt_effect, en_effect in list(self._pt_to_en_dict.items())[:1000]:
                effect_embedding = self._sentence_model.encode(self._normalize_text(pt_effect), convert_to_tensor=True)
                similarity = util.cos_sim(symptom_embedding, effect_embedding).item()
                if similarity > 0.5:
                    semantic_matches.append((en_effect, similarity))
            
            if semantic_matches:
                semantic_matches.sort(key=lambda x: x[1], reverse=True)
                return semantic_matches[0][0], semantic_matches[:5]
        
        return symptom, []
    
    def check_side_effect_similarity(self, user_effect: str, known_effects: List[str]) -> Tuple[List[SimilarEffect], str, List[Tuple[str, float]]]:
        """Verifica similaridade de efeitos colaterais"""
        if not known_effects:
            return [], user_effect, []
        
        translated_effect = user_effect
        possible_translations = []
        
        if self._pt_to_en_dict:
            translated_effect, possible_translations = self.translate_symptom_to_english(user_effect)
        
        if not self._sentence_model:
            return [], translated_effect, possible_translations
        
        user_embedding = self._sentence_model.encode(self._normalize_text(translated_effect), convert_to_tensor=True)
        similarities = []
        
        for effect in known_effects:
            if effect and str(effect) != 'nan':
                effect_embedding = self._sentence_model.encode(self._normalize_text(str(effect)), convert_to_tensor=True)
                similarity = util.cos_sim(user_embedding, effect_embedding).item()
                
                effect_pt = self._en_to_pt_dict.get(effect.lower(), effect) if self._en_to_pt_dict else effect
                
                similarities.append(SimilarEffect(
                    effect=effect,
                    similarity_score=similarity,
                    effect_pt=effect_pt if effect_pt != effect else None
                ))
        
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities, translated_effect, possible_translations
    
    def generate_basic_analysis(self, drug_name: str, user_effect: str, similar_effects: List[SimilarEffect], 
                               drug_info: DrugInfo, translated_effect: str = None, 
                               possible_translations: List[Tuple[str, float]] = None) -> str:
        """Gera análise básica baseada em dados"""
        translation_info = ""
        if translated_effect and translated_effect.lower() != user_effect.lower():
            translation_info = f"\n**Sintoma informado:** {user_effect}\n**Tradução detectada:** {translated_effect}\n"
            if possible_translations and len(possible_translations) > 1:
                translation_info += f"**Traduções alternativas consideradas:** {', '.join([t[0] for t in possible_translations[:3]])}\n"
        
        if not similar_effects:
            return f"""
**Análise:** Não encontrei o efeito '{user_effect}' como um efeito colateral documentado para {drug_name} em nossa base de dados.
{translation_info}
**Recomendação:** Isso não significa que o efeito não possa estar relacionado ao medicamento. Reações individuais podem variar. 
Recomendo consultar um médico ou farmacêutico para uma avaliação mais detalhada.

**⚠️ Importante:** Este sistema é apenas informativo e não substitui orientação médica profissional.
            """
        
        best_effect = similar_effects[0]
        similarity_score = best_effect.similarity_score
        best_match_display = best_effect.effect_pt or best_effect.effect
        
        if similarity_score > 0.8:
            confidence = "Alta"
            recommendation = "Este efeito está bem documentado para este medicamento. Se os sintomas persistirem ou piorarem, consulte um médico."
        elif similarity_score > 0.6:
            confidence = "Moderada"
            recommendation = "Existe uma possível relação com efeitos conhecidos. Monitore os sintomas e consulte um profissional de saúde se necessário."
        else:
            confidence = "Baixa"
            recommendation = "A relação com efeitos conhecidos é incerta. Recomendo consultar um médico para avaliação."
        
        return f"""
**Análise:** Encontrei uma correspondência com efeitos conhecidos de {drug_name}.
{translation_info}
**Efeito mais similar:** {best_match_display} (Similaridade: {similarity_score:.2f})
**Confiança da análise:** {confidence}

**Informações do medicamento:**
- Classe: {drug_info.drug_class or 'N/A'}
- Indicações: {drug_info.indications or 'N/A'}
- Severidade típica: {drug_info.side_effect_severity or 'N/A'}

**Recomendação:** {recommendation}

**⚠️ Importante:** Este sistema é apenas informativo e não substitui orientação médica profissional.
        """
    
    def generate_gemma_analysis(self, drug_name: str, user_symptom: str, similar_effects: List[SimilarEffect],
                               drug_info: DrugInfo, similarity_score: float = 0.0, 
                               translated_symptom: str = None) -> Optional[str]:
        """Gera análise usando GemmaMed"""
        if not self._gemma_client:
            return None
        
        matched_effects = [(effect.effect, effect.similarity_score) for effect in similar_effects[:5]]
        
        drug_info_dict = {
            'drug_class': drug_info.drug_class,
            'indications': drug_info.indications,
            'side_effect_severity': drug_info.side_effect_severity
        }
        
        return self._gemma_client.generate_enriched_analysis(
            drug_name=drug_name,
            user_symptom=user_symptom,
            matched_effects=matched_effects,
            drug_info=drug_info_dict,
            similarity_score=similarity_score,
            translated_symptom=translated_symptom
        )


_synmed_service = None

def get_synmed_service() -> SynMedService:
    """Obtém instância singleton do serviço SynMed"""
    global _synmed_service
    if _synmed_service is None:
        _synmed_service = SynMedService()
    return _synmed_service