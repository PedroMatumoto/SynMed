import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from typing import List, Tuple, Dict


@st.cache_resource
def load_model():
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")

@st.cache_data
def load_drug_data():
    try:
        drug_names = pd.read_csv('data/drug_names.csv', sep=';', header=None, names=['CID', 'drug_name'])
        
        realistic_drugs = pd.read_csv('data/realistic_drug_labels_side_effects.csv')
        
        sider_data = pd.read_csv('data/meddra_all_se.csv', sep=';', header=None, 
                                names=['CID', 'STITCH_ID', 'UMLS_ID', 'MedDRA_type', 'UMLS_ID2', 'side_effect'])
        
        return drug_names, realistic_drugs, sider_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def find_drug_matches(drug_input: str, drug_names_df: pd.DataFrame, realistic_drugs_df: pd.DataFrame) -> List[str]:
    normalized_input = normalize_text(drug_input)
    matches = []
    
    for _, row in drug_names_df.iterrows():
        if normalized_input in normalize_text(row['drug_name']):
            matches.append(row['drug_name'])
    
    for _, row in realistic_drugs_df.iterrows():
        if normalized_input in normalize_text(row['drug_name']):
            matches.append(row['drug_name'])
    
    return list(set(matches)) 

def get_side_effects_for_drug(drug_name: str, realistic_drugs_df: pd.DataFrame, sider_data_df: pd.DataFrame, drug_names_df: pd.DataFrame) -> List[str]:
    side_effects = []
    
    realistic_match = realistic_drugs_df[realistic_drugs_df['drug_name'].str.contains(drug_name, case=False, na=False)]
    if not realistic_match.empty:
        for _, row in realistic_match.iterrows():
            effects = str(row['side_effects']).split(', ')
            side_effects.extend([effect.strip() for effect in effects if effect.strip() != 'nan'])
    
    drug_cid = drug_names_df[drug_names_df['drug_name'].str.contains(drug_name, case=False, na=False)]
    if not drug_cid.empty:
        cid = drug_cid.iloc[0]['CID']
        sider_effects = sider_data_df[sider_data_df['CID'] == cid]['side_effect'].tolist()
        side_effects.extend(sider_effects)
    
    return list(set(side_effects))

def check_side_effect_similarity(user_effect: str, known_effects: List[str], model) -> List[Tuple[str, float]]:
    if not known_effects:
        return []
    
    user_embedding = model.encode(normalize_text(user_effect), convert_to_tensor=True)
    similarities = []
    
    for effect in known_effects:
        if effect and str(effect) != 'nan':
            effect_embedding = model.encode(normalize_text(str(effect)), convert_to_tensor=True)
            similarity = util.cos_sim(user_embedding, effect_embedding).item()
            similarities.append((effect, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def generate_ai_response(drug_name: str, user_effect: str, matches: List[Tuple[str, float]], drug_info: Dict) -> str:
    if not matches:
        return f"""
        **Análise:** Não encontrei o efeito '{user_effect}' como um efeito colateral documentado para {drug_name} em nossa base de dados.
        
        **Recomendação:** Isso não significa que o efeito não possa estar relacionado ao medicamento. Reações individuais podem variar. 
        Recomendo consultar um médico ou farmacêutico para uma avaliação mais detalhada.
        
        **⚠️ Importante:** Este sistema é apenas informativo e não substitui orientação médica profissional.
        """
    
    best_match = matches[0]
    similarity_score = best_match[1]
    
    if similarity_score > 0.8:
        confidence = "Alta"
        recommendation = "Este efeito está bem documentado para este medicamento. Se os sintomas persistirem ou piorarem, consulte um médico."
    elif similarity_score > 0.6:
        confidence = "Moderada"
        recommendation = "Existe uma possível relação com efeitos conhecidos. Monitore os sintomas e consulte um profissional de saúde se necessário."
    else:
        confidence = "Baixa"
        recommendation = "A relação com efeitos conhecidos é incerta. Recomendo consultar um médico para avaliação."
    
    response = f"""
    **Análise:** Encontrei uma correspondência com efeitos conhecidos de {drug_name}.
    
    **Efeito mais similar:** {best_match[0]} (Similaridade: {similarity_score:.2f})
    **Confiança da análise:** {confidence}
    
    **Informações do medicamento:**
    - Classe: {drug_info.get('drug_class', 'N/A')}
    - Indicações: {drug_info.get('indications', 'N/A')}
    - Severidade típica: {drug_info.get('side_effect_severity', 'N/A')}
    
    **Recomendação:** {recommendation}
    
    **⚠️ Importante:** Este sistema é apenas informativo e não substitui orientação médica profissional.
    """
    
    return response

sentence_model = load_model()
drug_names_df, realistic_drugs_df, sider_data_df = load_drug_data()

if drug_names_df is None:
    st.error("Erro ao carregar dados. Verifique se os arquivos CSV estão no diretório 'data/'.")
    st.stop()

st.title("💊 SynMed - Verificador de Efeitos Colaterais")
st.markdown("### Verifique se um sintoma pode ser efeito colateral de um medicamento")

col1, col2 = st.columns(2)

with col1:
    drug_input = st.text_input(
        "Nome do medicamento:",
        placeholder="Ex: Paracetamol, Ibuprofeno, Amoxicilina"
    )

with col2:
    effect_input = st.text_input(
        "Efeito/sintoma observado:",
        placeholder="Ex: dor de cabeça, náusea, tontura"
    )

if st.button("🔍 Verificar Efeito Colateral", type="primary"):
    if drug_input.strip() and effect_input.strip():
        with st.spinner("Analisando..."):
            drug_matches = find_drug_matches(drug_input, drug_names_df, realistic_drugs_df)
            
            if not drug_matches:
                st.warning(f"Medicamento '{drug_input}' não encontrado na base de dados.")
            else:
                selected_drug = drug_matches[0]
                
                if len(drug_matches) > 1:
                    st.info(f"Encontrados múltiplos medicamentos similares. Usando: {selected_drug}")
                    with st.expander("Ver todas as correspondências"):
                        for match in drug_matches:
                            st.write(f"- {match}")
                
                known_effects = get_side_effects_for_drug(selected_drug, realistic_drugs_df, sider_data_df, drug_names_df)
                
                similarities = check_side_effect_similarity(effect_input, known_effects, sentence_model)
                
                drug_info = {}
                drug_row = realistic_drugs_df[realistic_drugs_df['drug_name'].str.contains(selected_drug, case=False, na=False)]
                if not drug_row.empty:
                    drug_info = drug_row.iloc[0].to_dict()
                
                ai_response = generate_ai_response(selected_drug, effect_input, similarities, drug_info)
                
                st.markdown("---")
                st.markdown("## 🤖 Análise AI")
                st.markdown(ai_response)
                
                if similarities:
                    st.markdown("### 📊 Efeitos Colaterais Similares Conhecidos")
                    for i, (effect, score) in enumerate(similarities[:5]):
                        confidence_color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                        st.write(f"{confidence_color} **{effect}** - Similaridade: {score:.3f}")
                
                if known_effects:
                    with st.expander(f"Ver todos os efeitos colaterais conhecidos de {selected_drug}"):
                        for effect in sorted(set(known_effects)):
                            if effect and str(effect) != 'nan':
                                st.write(f"• {effect}")
    else:
        st.warning("Por favor, preencha tanto o nome do medicamento quanto o efeito observado.")

st.markdown("---")
st.markdown("""
**⚠️ AVISO IMPORTANTE:**
- Este sistema é apenas informativo e educacional
- NÃO substitui consulta médica ou farmacêutica
- Em caso de efeitos adversos graves, procure atendimento médico imediatamente
- Sempre consulte profissionais de saúde antes de tomar decisões sobre medicamentos
""")
