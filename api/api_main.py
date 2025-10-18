import re
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
import motor.motor_asyncio
from .api_models import (
    User, UserCreate, UserLogin, UserResponse, Token, SearchHistory,
    DrugEffectRequest, DrugEffectResponse, SimilarEffect, DrugInfo,
    EffectAnalysisHistory, DrugSearchResult, ExtractedSymptom
)
from .api_auth import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    get_current_active_user,
    get_current_admin_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user_optional,
)
from .synmed_service import get_synmed_service
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os
from typing import List, Optional

# Configuração de quota gratuita para usuários anônimos
FREE_REQUESTS_LIMIT = int(os.getenv("FREE_REQUESTS_LIMIT", "10"))

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar MongoDB e Beanie
    client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    await init_beanie(
        database=client.synmed, 
        document_models=[User, SearchHistory, EffectAnalysisHistory]
    )
    yield

app = FastAPI(
    title="SynMed API",
    description="API para verificação de efeitos colaterais de medicamentos",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

synmed_service = get_synmed_service()


@app.post("/auth/register", response_model=UserResponse, status_code=201)
async def register_user(user_data: UserCreate):
    """Registra um novo usuário"""
    existing_user = await User.find_one(User.email == user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=400, 
            detail="Email already registered"
        )
    
    if not user_data.email or not user_data.password or user_data.email.strip() == "" or user_data.password.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Email and password cannot be empty"
        )
    
    pattern = "^(?=.*[0-9])(?=.*[^A-Za-z0-9]).{6,}$"
    if not re.match(pattern, user_data.password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters long and contain at least one number and one special character"
        )
    
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    await user.create()
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_admin=getattr(user, "is_admin", False),
        created_at=user.created_at
    )


@app.post("/auth/login", response_model=Token)
async def login_user(user_data: UserLogin):
    """Autentica usuário e retorna token"""
    user = await authenticate_user(user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Obtém informações do usuário atual"""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_admin=getattr(current_user, "is_admin", False),
        created_at=current_user.created_at
    )


@app.get("/admin/users", response_model=List[UserResponse])
async def admin_list_users(admin_user: User = Depends(get_current_admin_user)):
    """Lista todos os usuários (somente admin)"""
    users = await User.find_all().to_list()
    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            full_name=u.full_name,
            is_active=u.is_active,
            is_admin=getattr(u, "is_admin", False),
            created_at=u.created_at,
        )
        for u in users
    ]


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: str, admin_user: User = Depends(get_current_admin_user)):
    user = await User.get(user_id)
    if not user:
        user = await User.find_one(User.id == user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await user.delete()
    return {"message": "User deleted successfully"}


@app.get("/drugs/search")
async def search_drugs(
    query: str,
    use_semantic: bool = True,
    threshold: float = 0.6,
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> List[DrugSearchResult]:
    """Busca medicamentos por nome"""
    try:
        return synmed_service.find_drug_matches(query, use_semantic, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca: {str(e)}")


@app.get("/drugs/{drug_name}/effects")
async def get_drug_effects(
    drug_name: str,
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> List[str]:
    """Obtém efeitos colaterais conhecidos de um medicamento"""
    try:
        effects = synmed_service.get_side_effects_for_drug(drug_name)
        return effects
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter efeitos: {str(e)}")


class ExtractSymptomsRequest(BaseModel):
    text: str


@app.post("/extract-symptoms")
async def extract_symptoms(
    request: ExtractSymptomsRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> List[ExtractedSymptom]:
    """Extrai sintomas de um texto em linguagem natural"""
    try:
        extracted = synmed_service.extract_symptoms_from_text(request.text)
        return extracted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na extração: {str(e)}")


@app.post("/analyze-effect", response_model=DrugEffectResponse)
async def analyze_drug_effect(
    request: Request,
    analysis_request: DrugEffectRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Analisa se um ou múltiplos sintomas podem ser efeitos colaterais de um medicamento"""
    
    client_ip = request.client.host if request.client else "unknown"
    
    if current_user is None:
        anon_id = f"anonymous:{client_ip}"
        anon_count = await EffectAnalysisHistory.find(
            EffectAnalysisHistory.user_id == anon_id
        ).count()
        if anon_count >= FREE_REQUESTS_LIMIT:
            raise HTTPException(
                status_code=401, 
                detail=f"Free request quota exceeded ({FREE_REQUESTS_LIMIT} analyses). Please register or login."
            )
    
    try:
        drug_matches = synmed_service.find_drug_matches(
            analysis_request.drug_name,
            analysis_request.use_semantic_search,
            analysis_request.similarity_threshold
        )
        
        if not drug_matches:
            raise HTTPException(
                status_code=404,
                detail=f"Medicamento '{analysis_request.drug_name}' não encontrado na base de dados"
            )
        
        selected_drug = drug_matches[0].drug_name
        known_effects = synmed_service.get_side_effects_for_drug(selected_drug)
        drug_info = synmed_service.get_drug_info(selected_drug)
        
        symptoms_to_analyze = []
        extracted_symptoms = None
        natural_language_input = None
        
        if analysis_request.natural_language_text and analysis_request.natural_language_text.strip():
            natural_language_input = analysis_request.natural_language_text.strip()
            extracted_symptoms = synmed_service.extract_symptoms_from_text(natural_language_input)
            symptoms_to_analyze = [symptom.text for symptom in extracted_symptoms]
            
        elif analysis_request.symptoms:
            symptoms_to_analyze = [s.strip() for s in analysis_request.symptoms if s.strip()]
            
        else:
            symptoms_to_analyze = [analysis_request.effect_symptom.strip()]
        
        if not symptoms_to_analyze:
            raise HTTPException(
                status_code=400,
                detail="Nenhum sintoma válido foi fornecido para análise"
            )
        
        if len(symptoms_to_analyze) > 1:
            symptom_analyses, overall_confidence, avg_similarity = synmed_service.analyze_multiple_symptoms(
                selected_drug, symptoms_to_analyze, known_effects
            )
            
            basic_analysis = synmed_service.generate_multi_symptom_analysis(
                selected_drug, symptom_analyses, drug_info, overall_confidence
            )
            
            gemma_analysis = None
            if analysis_request.use_gemma_analysis:
                gemma_analysis = synmed_service.generate_multi_symptom_gemma_analysis(
                    selected_drug, symptom_analyses, drug_info, overall_confidence
                )
            
            first_symptom = symptoms_to_analyze[0]
            first_analysis = symptom_analyses[0] if symptom_analyses else None
            
            user_id = str(current_user.id) if current_user else f"anonymous:{client_ip}"
            history_entry = EffectAnalysisHistory(
                user_id=user_id,
                drug_name=selected_drug,
                effect_symptom=first_symptom,  # Para compatibilidade
                symptoms=symptoms_to_analyze,
                extracted_symptoms=extracted_symptoms,
                natural_language_input=natural_language_input,
                confidence=overall_confidence,
                similarity_score=avg_similarity,
                overall_confidence=overall_confidence,
                use_semantic_search=analysis_request.use_semantic_search,
                use_gemma_analysis=analysis_request.use_gemma_analysis
            )
            await history_entry.create()
            
            return DrugEffectResponse(
                drug_name=selected_drug,
                user_effect=first_symptom,  # Para compatibilidade
                translated_effect=first_analysis.translated_symptom if first_analysis else None,
                possible_translations=None,
                similar_effects=first_analysis.similar_effects if first_analysis else [],
                confidence=overall_confidence,
                similarity_score=avg_similarity,
                drug_info=drug_info,
                basic_analysis=basic_analysis,
                gemma_analysis=gemma_analysis,
                all_known_effects=known_effects,
                extracted_symptoms=extracted_symptoms,
                symptom_analyses=symptom_analyses,
                overall_confidence=overall_confidence
            )
        
        else:
            single_symptom = symptoms_to_analyze[0]
            
            similar_effects, translated_effect, possible_translations = synmed_service.check_side_effect_similarity(
                single_symptom, known_effects
            )
            
            similarity_score = similar_effects[0].similarity_score if similar_effects else 0.0
            if similarity_score > 0.8:
                confidence = "Alta"
            elif similarity_score > 0.6:
                confidence = "Moderada"
            else:
                confidence = "Baixa"
            
            basic_analysis = synmed_service.generate_basic_analysis(
                selected_drug, single_symptom, similar_effects,
                drug_info, translated_effect, possible_translations
            )
            
            gemma_analysis = None
            if analysis_request.use_gemma_analysis:
                gemma_analysis = synmed_service.generate_gemma_analysis(
                    selected_drug, single_symptom, similar_effects,
                    drug_info, similarity_score, translated_effect
                )
            
            user_id = str(current_user.id) if current_user else f"anonymous:{client_ip}"
            history_entry = EffectAnalysisHistory(
                user_id=user_id,
                drug_name=selected_drug,
                effect_symptom=single_symptom,
                symptoms=[single_symptom],
                extracted_symptoms=extracted_symptoms,
                natural_language_input=natural_language_input,
                confidence=confidence,
                similarity_score=similarity_score,
                overall_confidence=confidence,
                use_semantic_search=analysis_request.use_semantic_search,
                use_gemma_analysis=analysis_request.use_gemma_analysis
            )
            await history_entry.create()
            
            return DrugEffectResponse(
                drug_name=selected_drug,
                user_effect=single_symptom,
                translated_effect=translated_effect if translated_effect != single_symptom else None,
                possible_translations=possible_translations if possible_translations else None,
                similar_effects=similar_effects,
                confidence=confidence,
                similarity_score=similarity_score,
                drug_info=drug_info,
                basic_analysis=basic_analysis,
                gemma_analysis=gemma_analysis,
                all_known_effects=known_effects,
                extracted_symptoms=extracted_symptoms,
                symptom_analyses=None,
                overall_confidence=confidence
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")


@app.get("/my-analyses", response_model=List[EffectAnalysisHistory])
async def get_my_analysis_history(
    current_user: User = Depends(get_current_active_user),
    limit: int = 50,
    skip: int = 0
):
    """Obtém o histórico de análises do usuário atual"""
    if limit > 100:
        limit = 100 
    
    analysis_history = await EffectAnalysisHistory.find(
        EffectAnalysisHistory.user_id == str(current_user.id)
    ).sort(-EffectAnalysisHistory.timestamp).skip(skip).limit(limit).to_list()
    
    return analysis_history


@app.delete("/my-analyses/{analysis_id}")
async def delete_analysis_history(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Remove uma análise específica do histórico"""
    analysis = await EffectAnalysisHistory.find_one(
        EffectAnalysisHistory.id == analysis_id,
        EffectAnalysisHistory.user_id == str(current_user.id)
    )
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis history not found")
    
    await analysis.delete()
    return {"message": "Analysis history deleted successfully"}


@app.delete("/my-analyses")
async def clear_analysis_history(current_user: User = Depends(get_current_active_user)):
    """Limpa todo o histórico de análises do usuário"""
    await EffectAnalysisHistory.find(EffectAnalysisHistory.user_id == str(current_user.id)).delete()
    return {"message": "Analysis history cleared successfully"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "SynMed API"
    }


@app.get("/")
async def root():
    return {
        "message": "SynMed API - Sistema de Verificação de Efeitos Colaterais",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)