from datetime import datetime
from beanie import Document
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from pymongo import IndexModel


class User(Document):
    email: EmailStr
    hashed_password: str
    full_name: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = datetime.now()

    class Settings:
        name = "users"
        indexes = [
            IndexModel("email", unique=True),
        ]


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    is_active: bool
    is_admin: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class SearchHistory(Document):
    user_id: str
    search_params: Dict[str, Any]
    results_count: int
    timestamp: datetime = datetime.utcnow()

    class Settings:
        name = "search_history"
        indexes = [
            IndexModel("user_id"),
            IndexModel("timestamp"),
        ]


# Modelos específicos do SynMed
class DrugEffectRequest(BaseModel):
    drug_name: str
    effect_symptom: str
    symptoms: Optional[List[str]] = None
    natural_language_text: Optional[str] = None
    use_semantic_search: bool = True
    similarity_threshold: float = 0.6
    use_gemma_analysis: bool = True


class SimilarEffect(BaseModel):
    effect: str
    similarity_score: float
    effect_pt: Optional[str] = None


class SymptomAnalysis(BaseModel):
    original_symptom: str
    translated_symptom: Optional[str] = None
    similar_effects: List[SimilarEffect]
    confidence: str
    similarity_score: float
    
    
class ExtractedSymptom(BaseModel):
    text: str
    confidence_score: float
    source_span: str 


class DrugInfo(BaseModel):
    drug_class: Optional[str] = None
    indications: Optional[str] = None
    side_effect_severity: Optional[str] = None


class DrugEffectResponse(BaseModel):
    drug_name: str
    user_effect: str
    translated_effect: Optional[str] = None
    possible_translations: Optional[List[Dict[str, float]]] = None
    similar_effects: List[SimilarEffect]
    confidence: str
    similarity_score: float
    drug_info: DrugInfo
    basic_analysis: str
    gemma_analysis: Optional[str] = None
    all_known_effects: Optional[List[str]] = None
    extracted_symptoms: Optional[List[ExtractedSymptom]] = None
    symptom_analyses: Optional[List[SymptomAnalysis]] = None
    overall_confidence: Optional[str] = None  # Confiança considerando todos os sintomas


class DrugSearchResult(BaseModel):
    drug_name: str
    similarity_score: Optional[float] = None
    source: str  # "official" or "realistic"


class EffectAnalysisHistory(Document):
    user_id: str
    drug_name: str
    effect_symptom: str
    symptoms: Optional[List[str]] = None
    extracted_symptoms: Optional[List[ExtractedSymptom]] = None
    natural_language_input: Optional[str] = None
    confidence: str
    similarity_score: float
    overall_confidence: Optional[str] = None
    use_semantic_search: bool
    use_gemma_analysis: bool
    timestamp: datetime = datetime.utcnow()

    class Settings:
        name = "effect_analysis_history"
        indexes = [
            IndexModel("user_id"),
            IndexModel("timestamp"),
            IndexModel("drug_name"),
        ]