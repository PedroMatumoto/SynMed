import requests
import streamlit as st
from typing import Optional, Dict
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

class GemmaMedClient:
    """Cliente para interagir com o modelo GemmaMed-27B-IT via Hugging Face Inference Endpoint"""
    
    def __init__(self, endpoint_url: str, api_token: Optional[str] = None):
        """
        Args:
            endpoint_url: URL do inference endpoint
            api_token: Token de API da Hugging Face (opcional, use secrets do Streamlit)
        """
        self.endpoint_url = endpoint_url
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Configurar sessão com retry automático
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            backoff_factor=2  # 2, 4, 8 segundos
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def check_endpoint_status(self) -> Dict[str, any]:
        """
        Verifica o status do endpoint antes de fazer requisições
        """
        try:
            # Tentar uma requisição simples para verificar disponibilidade
            response = requests.get(
                self.endpoint_url.replace('/v1/chat/completions', '').replace('/generate', ''),
                headers={"Authorization": f"Bearer {self.api_token}"},
                timeout=5
            )
            return {
                "available": response.status_code != 503,
                "status_code": response.status_code,
                "message": "Endpoint disponível" if response.status_code != 503 else "Endpoint indisponível"
            }
        except Exception as e:
            return {
                "available": False,
                "status_code": None,
                "message": f"Erro ao verificar endpoint: {str(e)}"
            }
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        wait_for_model: bool = True
    ) -> Optional[str]:
        """
        Gera resposta usando o modelo GemmaMed
        
        Args:
            prompt: Prompt para o modelo
            max_new_tokens: Número máximo de tokens a gerar
            temperature: Controla aleatoriedade (0.0 = determinístico, 1.0 = criativo)
            top_p: Nucleus sampling
            do_sample: Se deve usar amostragem
            wait_for_model: Se deve aguardar o modelo carregar (importante para endpoints privados)
            
        Returns:
            Resposta gerada pelo modelo ou None em caso de erro
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": wait_for_model,
                "use_cache": False
            }
        }
        
        try:
            with st.spinner("🤖 Consultando modelo médico especializado (GemmaMed)..."):
                response = self.session.post(
                    self.endpoint_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60  # Aumentar timeout para 60 segundos
                )
                
                # Tratamento específico para erro 503
                if response.status_code == 503:
                    error_data = response.json() if response.content else {}
                    estimated_time = error_data.get('estimated_time', 20)
                    
                    st.warning(f"⏳ O modelo está carregando. Tempo estimado: {estimated_time} segundos. Aguardando...")
                    
                    # Aguardar e tentar novamente
                    time.sleep(min(estimated_time + 5, 30))  # Aguardar no máximo 30 segundos
                    
                    response = self.session.post(
                        self.endpoint_url,
                        headers=self.headers,
                        json=payload,
                        timeout=60
                    )
                
                response.raise_for_status()
                
                result = response.json()
                
                # Diferentes formatos de resposta possíveis
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "generated_text" in result[0]:
                        return result[0]["generated_text"]
                    elif isinstance(result[0], str):
                        return result[0]
                elif isinstance(result, dict):
                    if "generated_text" in result:
                        return result["generated_text"]
                    elif "error" in result:
                        st.error(f"❌ Erro do modelo: {result['error']}")
                        return None
                
                return None
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout ao conectar com o modelo. O endpoint pode estar sobrecarregado.")
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                st.error("🔴 **Endpoint Indisponível (503)**\n\n"
                        "O endpoint do modelo está temporariamente indisponível. Possíveis causas:\n"
                        "- O endpoint está em modo sleep (inativo por falta de uso)\n"
                        "- O modelo está carregando\n"
                        "- Limite de requisições atingido\n\n"
                        "**Soluções:**\n"
                        "1. Aguarde alguns minutos e tente novamente\n"
                        "2. Acesse o [Hugging Face](https://huggingface.co/endpoints) e verifique o status do endpoint\n"
                        "3. Use a análise baseada em dados (desabilite GemmaMed)")
            elif e.response.status_code == 401:
                st.error("🔐 **Erro de Autenticação (401)**\n\n"
                        "Token de API inválido ou expirado.\n"
                        "Verifique o token em `.streamlit/secrets.toml`")
            elif e.response.status_code == 429:
                st.error("⚠️ **Limite de Taxa Excedido (429)**\n\n"
                        "Muitas requisições em curto período. Aguarde alguns minutos.")
            else:
                st.error(f"❌ Erro HTTP {e.response.status_code}: {str(e)}")
            return None
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erro de rede ao conectar com o modelo: {str(e)}")
            return None
            
        except Exception as e:
            st.error(f"❌ Erro inesperado: {str(e)}")
            return None

    def create_medical_prompt(
        self,
        drug_name: str,
        user_symptom: str,
        matched_effects: list,
        drug_info: Dict,
        similarity_score: float = 0.0,
        translated_symptom: str = None
    ) -> str:
        """
        Cria um prompt otimizado para análise médica de efeitos colaterais
        """
        # Preparar informações dos efeitos similares
        similar_effects_text = ""
        if matched_effects:
            similar_effects_text = "\n".join([
                f"- {effect} (similaridade: {score:.2f})"
                for effect, score in matched_effects[:5]
            ])
        
        # Informações do medicamento
        drug_class = drug_info.get('drug_class', 'Não especificado')
        indications = drug_info.get('indications', 'Não especificado')
        severity = drug_info.get('side_effect_severity', 'Não especificado')
        
        # Construir prompt estruturado
        prompt = f"""<start_of_turn>user
Você é um assistente médico especializado em farmacologia e efeitos colaterais de medicamentos.

INFORMAÇÕES DO CASO:
- Medicamento: {drug_name}
- Classe: {drug_class}
- Indicações: {indications}
- Sintoma relatado pelo paciente: {user_symptom}"""

        if translated_symptom and translated_symptom != user_symptom:
            prompt += f"\n- Sintoma traduzido: {translated_symptom}"
        
        if similar_effects_text:
            prompt += f"\n\nEFEITOS COLATERAIS CONHECIDOS SIMILARES:\n{similar_effects_text}"
            prompt += f"\n\nGrau de similaridade mais alto: {similarity_score:.2f}"
        
        prompt += f"\n\nSeveridade típica dos efeitos: {severity}"
        
        prompt += """\n\nCom base nessas informações, forneça uma análise médica detalhada e estruturada incluindo:

1. **Avaliação da Relação**: Avalie a probabilidade de o sintoma relatado ser um efeito colateral do medicamento (Alta/Média/Baixa)
2. **Mecanismo**: Explique brevemente o possível mecanismo farmacológico
3. **Gravidade**: Classifique a gravidade do efeito (leve, moderado, grave)
4. **Recomendações Práticas**: O que o paciente deve fazer
5. **Sinais de Alerta**: Quando buscar atendimento médico urgente

Seja claro, objetivo e acessível para o público leigo.
<end_of_turn>
<start_of_turn>model
"""

        return prompt

    def generate_enriched_analysis(
        self,
        drug_name: str,
        user_symptom: str,
        matched_effects: list,
        drug_info: Dict,
        similarity_score: float = 0.0,
        translated_symptom: str = None
    ) -> Optional[str]:
        """
        Gera análise enriquecida usando o modelo GemmaMed
        """
        prompt = self.create_medical_prompt(
            drug_name=drug_name,
            user_symptom=user_symptom,
            matched_effects=matched_effects,
            drug_info=drug_info,
            similarity_score=similarity_score,
            translated_symptom=translated_symptom
        )
        
        return self.generate_response(
            prompt=prompt,
            max_new_tokens=700,
            temperature=0.7,
            top_p=0.95,
            wait_for_model=True  # Importante para endpoints privados
        )

@st.cache_resource
def get_gemma_client():
    """Inicializa e cacheia o cliente GemmaMed"""
    endpoint_url = os.getenv("HUGGINGFACE_MODEL_API_TOKEN", "")
    return GemmaMedClient(endpoint_url)

def check_gemma_availability() -> bool:
    """
    Verifica se o endpoint GemmaMed está disponível
    """
    try:
        client = get_gemma_client()
        status = client.check_endpoint_status()
        return status["available"]
    except:
        return False