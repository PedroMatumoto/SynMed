import re
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from .api_models import ExtractedSymptom

try:
    import spacy
except ImportError:
    spacy = None


class SymptomExtractor:
    """Serviço para extrair sintomas de texto em linguagem natural"""
    
    def __init__(self):
        self._nlp = None
        self._symptom_model = None
        self._known_symptoms = []
        self._portuguese_symptoms = []
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa modelos de NLP e dados de sintomas"""
        try:
            if spacy is not None:
                try:
                    self._nlp = spacy.load("pt_core_news_sm")
                except OSError:
                    try:
                        self._nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        print("Aviso: Nenhum modelo spaCy encontrado. Funcionalidade de NLP limitada.")
                        self._nlp = None
            else:
                print("Aviso: spaCy não está instalado. Funcionalidade de NLP limitada.")
                self._nlp = None
            
            self._symptom_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
            
            self._load_symptom_database()
            
        except Exception as e:
            print(f"Erro ao inicializar SymptomExtractor: {e}")
    
    def _load_symptom_database(self):
        """Carrega base de dados de sintomas conhecidos"""
        try:
            translations_df = pd.read_csv('data/side_effects_translated.csv')
            
            self._known_symptoms = translations_df['Original'].str.lower().unique().tolist()
            self._portuguese_symptoms = translations_df['Traduzido'].str.lower().unique().tolist()
            
            common_pt_symptoms = [
                'dor de cabeça', 'cefaleia', 'enxaqueca', 'dor no peito', 'dor abdominal',
                'náusea', 'enjôo', 'vômito', 'tontura', 'vertigem', 'sonolência',
                'insônia', 'fadiga', 'cansaço', 'fraqueza', 'febre', 'calafrios',
                'suor', 'transpiração', 'palpitação', 'taquicardia', 'bradicardia',
                'falta de ar', 'dispneia', 'tosse', 'dor muscular', 'mialgia',
                'dor articular', 'artralgia', 'inchaço', 'edema', 'coceira',
                'prurido', 'erupção', 'vermelhidão', 'constipação', 'prisão de ventre',
                'diarreia', 'dor estomacal', 'azia', 'queimação', 'boca seca',
                'xerostomia', 'visão turva', 'diplopia', 'zumbido no ouvido',
                'tinnitus', 'perda de apetite', 'anorexia', 'ganho de peso',
                'perda de peso', 'ansiedade', 'depressão', 'irritabilidade',
                'confusão mental', 'esquecimento', 'concentração diminuída',
                'tremor', 'espasmos', 'câimbras', 'formigamento', 'parestesia',
                'dormência', 'queimação na pele', 'sensibilidade', 'hipertensão',
                'pressão alta', 'hipotensão', 'pressão baixa', 'arritmia'
            ]
            
            additional_en_symptoms = [
                'headache', 'chest pain', 'abdominal pain', 'stomach pain',
                'nausea', 'vomiting', 'dizziness', 'drowsiness', 'insomnia',
                'fatigue', 'weakness', 'fever', 'chills', 'sweating', 'palpitations',
                'shortness of breath', 'cough', 'muscle pain', 'joint pain',
                'swelling', 'itching', 'rash', 'redness', 'constipation', 'diarrhea',
                'heartburn', 'dry mouth', 'blurred vision', 'ringing in ears',
                'loss of appetite', 'weight gain', 'weight loss', 'anxiety',
                'depression', 'irritability', 'confusion', 'memory problems',
                'concentration problems', 'tremor', 'spasms', 'cramps',
                'tingling', 'numbness', 'burning sensation', 'sensitivity',
                'high blood pressure', 'low blood pressure', 'irregular heartbeat',
                'back pain', 'leg pain', 'arm pain', 'stomach ache',
                'indigestion', 'gas', 'bloating', 'burping', 'acid reflux',
                'mal estar', 'mal-estar', 'indisposição', 'enjoo', 'ânsia',
                'dor nas costas', 'dor nas pernas', 'dor nos braços', 'dor na barriga',
                'dor de estômago', 'gastura', 'gases', 'flatulência', 'arrotos',
                'indigestão', 'empachamento', 'batedeira', 'coração acelerado',
                'tonteira', 'zonzeira', 'vista embaçada', 'visão embaçada',
                'coceira na pele', 'comichão', 'ardência', 'formigamento nas mãos',
                'formigamento nos pés', 'dormência nas mãos', 'dormência nos pés'
            ]
            
            self._portuguese_symptoms.extend(common_pt_symptoms)
            self._known_symptoms.extend(additional_en_symptoms)
            
            self._known_symptoms = list(set([s.lower().strip() for s in self._known_symptoms if s and str(s) != 'nan' and len(s.strip()) > 2]))
            self._portuguese_symptoms = list(set([s.lower().strip() for s in self._portuguese_symptoms if s and str(s) != 'nan' and len(s.strip()) > 2]))
            
            print(f"✅ Carregados {len(self._known_symptoms)} sintomas em inglês e {len(self._portuguese_symptoms)} sintomas em português")
            
        except Exception as e:
            print(f"Erro ao carregar base de sintomas: {e}")
            self._portuguese_symptoms = [
                'dor de cabeça', 'náusea', 'tontura', 'fadiga', 'febre',
                'dor no peito', 'falta de ar', 'tosse', 'vômito', 'diarreia'
            ]
            self._known_symptoms = [
                'headache', 'nausea', 'dizziness', 'fatigue', 'fever',
                'chest pain', 'shortness of breath', 'cough', 'vomiting', 'diarrhea'
            ]
    
    def extract_symptoms_from_text(self, text: str, confidence_threshold: float = 0.6) -> List[ExtractedSymptom]:
        """Extrai sintomas de um texto em linguagem natural"""
        if not text or not text.strip():
            return []
        
        extracted = []
        text_lower = text.lower().strip()
        
        all_symptoms = self._portuguese_symptoms + self._known_symptoms
        for symptom in all_symptoms:
            if len(symptom) < 3:
                continue
                
            if symptom in text_lower:
                start_idx = 0
                while True:
                    idx = text_lower.find(symptom, start_idx)
                    if idx == -1:
                        break
                    
                    if (idx == 0 or not text_lower[idx-1].isalnum()) and \
                       (idx + len(symptom) >= len(text_lower) or not text_lower[idx + len(symptom)].isalnum()):
                        
                        source_span = text[idx:idx + len(symptom)]
                        
                        if not any(existing.text.lower() == symptom for existing in extracted):
                            extracted.append(ExtractedSymptom(
                                text=symptom,
                                confidence_score=1.0,
                                source_span=source_span
                            ))
                    
                    start_idx = idx + 1
        
        if self._symptom_model and len(extracted) < 5: 
            semantic_matches = self._extract_semantic_symptoms(text, confidence_threshold)
            
            for match in semantic_matches:
                if not any(existing.text.lower() == match.text.lower() for existing in extracted):
                    extracted.append(match)
        
        if self._nlp and len(extracted) < 10:
            nlp_matches = self._extract_nlp_symptoms(text, confidence_threshold)
            
            for match in nlp_matches:
                if not any(existing.text.lower() == match.text.lower() for existing in extracted):
                    extracted.append(match)
        
        extracted.sort(key=lambda x: x.confidence_score, reverse=True)
        return extracted[:15]  
    
    def _extract_semantic_symptoms(self, text: str, threshold: float) -> List[ExtractedSymptom]:
        """Extrai sintomas usando busca semântica"""
        if not self._symptom_model:
            return []
        
        matches = []
        
        try:
            sentences = self._split_into_fragments(text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 5:
                    continue
                
                sentence_embedding = self._symptom_model.encode(sentence.lower(), convert_to_tensor=True)
                
                best_matches = []
                all_symptoms = list(set(self._portuguese_symptoms + self._known_symptoms))
                
                for symptom in all_symptoms[:500]:
                    if len(symptom) < 3:
                        continue
                        
                    symptom_embedding = self._symptom_model.encode(symptom, convert_to_tensor=True)
                    similarity = util.cos_sim(sentence_embedding, symptom_embedding).item()
                    
                    if similarity >= threshold:
                        best_matches.append((symptom, similarity, sentence))
                
                best_matches.sort(key=lambda x: x[1], reverse=True)
                
                for symptom, score, source in best_matches[:3]:
                    matches.append(ExtractedSymptom(
                        text=symptom,
                        confidence_score=score,
                        source_span=source
                    ))
        
        except Exception as e:
            print(f"Erro na extração semântica: {e}")
        
        return matches
    
    def _extract_nlp_symptoms(self, text: str, threshold: float) -> List[ExtractedSymptom]:
        """Extrai sintomas usando processamento de linguagem natural com spaCy"""
        if not self._nlp:
            return []
        
        matches = []
        
        try:
            doc = self._nlp(text)
            
            symptom_patterns = [
                r'(?:sinto|senti|estou\s+sentindo|tenho|tive|estou\s+com)\s+([^.!?]+)',
                r'dor\s+(?:na|no|em|nas|nos)\s+([^.!?]+)',
                r'(?:me\s+)?(?:dói|dóem)\s+([^.!?]+)',
                r'([^.!?]+)\s+(?:forte|intensa?|severa?|constante)',
                r'(?:após|depois\s+de)\s+(?:tomar|usar)\s+[^,]+,?\s*(?:senti|tive|apareceu)\s+([^.!?]+)',
            ]
            
            for pattern in symptom_patterns:
                for match in re.finditer(pattern, text.lower()):
                    potential_symptom = match.group(1).strip()
                    
                    if len(potential_symptom) > 2 and len(potential_symptom) < 50:
                        if self._is_likely_symptom(potential_symptom):
                            matches.append(ExtractedSymptom(
                                text=potential_symptom,
                                confidence_score=0.7,
                                source_span=match.group(0)
                            ))
            
            for ent in doc.ents:
                if ent.label_ in ['MISC', 'ORG'] and self._is_likely_symptom(ent.text.lower()):
                    matches.append(ExtractedSymptom(
                        text=ent.text.lower(),
                        confidence_score=0.6,
                        source_span=ent.text
                    ))
        
        except Exception as e:
            print(f"Erro na extração NLP: {e}")
        
        return matches
    
    def _split_into_fragments(self, text: str) -> List[str]:
        """Divide texto em fragmentos menores para análise"""
        fragments = re.split(r'[.!?;,]\s*(?:e|mas|porém|então|depois|além disso)?\s*', text)
        
        fragments = [f.strip() for f in fragments if len(f.strip()) > 5]
        
        if not fragments:
            fragments = [text]
        
        return fragments
    
    def _is_likely_symptom(self, text: str) -> bool:
        """Verifica se um texto provavelmente representa um sintoma"""
        text_lower = text.lower().strip()
        
        symptom_indicators = [
            'dor', 'ache', 'pain', 'náusea', 'nausea', 'tontura', 'dizz',
            'febre', 'fever', 'cansaço', 'fadiga', 'fatigue', 'tired',
            'fraqueza', 'weakness', 'vômito', 'vomit', 'enjôo',
            'coceira', 'itch', 'vermelhidão', 'redness', 'inchaço', 'swell',
            'falta', 'shortness', 'difficulty', 'dificuldade', 'burning',
            'queimação', 'formigamento', 'tingling', 'dormência', 'numbness'
        ]
        
        exclude_words = [
            'medicamento', 'remédio', 'comprimido', 'pílula', 'dose',
            'médico', 'hospital', 'farmácia', 'receita', 'prescrição',
            'tomar', 'usar', 'aplicar', 'ingerir'
        ]
        
        has_symptom_indicator = any(indicator in text_lower for indicator in symptom_indicators)
        
        has_exclude_word = any(exclude in text_lower for exclude in exclude_words)
        
        reasonable_length = 3 <= len(text_lower) <= 50
        
        return has_symptom_indicator and not has_exclude_word and reasonable_length
    
    def validate_symptoms(self, symptoms: List[str]) -> List[str]:
        """Valida e limpa uma lista de sintomas"""
        validated = []
        
        for symptom in symptoms:
            clean_symptom = symptom.strip().lower()
            
            if len(clean_symptom) >= 3 and self._is_likely_symptom(clean_symptom):
                validated.append(clean_symptom)
        
        return list(set(validated))

_symptom_extractor = None


def get_symptom_extractor() -> SymptomExtractor:
    """Obtém instância singleton do extrator de sintomas"""
    global _symptom_extractor
    if _symptom_extractor is None:
        _symptom_extractor = SymptomExtractor()
    return _symptom_extractor