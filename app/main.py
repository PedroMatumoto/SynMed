import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from typing import List, Tuple, Dict

from data_op import treat_realistic_drugs, semantic_drug_search, load_drug_search_model


@st.cache_resource
def load_model():
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")

@st.cache_resource  
def load_drug_model():
    return load_drug_search_model()

@st.cache_data
def load_drug_data():
    try:
        drug_names = pd.read_csv('data/drug_names.csv', sep=';', header=None, names=['CID', 'drug_name'])
        
        realistic_drugs = pd.read_csv('data/realistic_drug_labels_side_effects.csv')
        treated_realistic_drugs = treat_realistic_drugs(realistic_drugs)
        
        sider_data = pd.read_csv('data/meddra_all_se.csv', sep=';', header=None, 
                                names=['CID', 'STITCH_ID', 'UMLS_ID', 'MedDRA_type', 'UMLS_ID2', 'side_effect'])

        return drug_names, treated_realistic_drugs, sider_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_translations():
    """Carrega o arquivo de traduções de efeitos colaterais"""
    try:
        translations_df = pd.read_csv('data/side_effects_translated.csv')
        # Criar dicionário bidirecional para tradução
        pt_to_en = dict(zip(translations_df['Traduzido'].str.lower(), translations_df['Original']))
        en_to_pt = dict(zip(translations_df['Original'].str.lower(), translations_df['Traduzido']))
        return pt_to_en, en_to_pt
    except Exception as e:
        st.warning(f"Aviso: Não foi possível carregar traduções. Usando modo inglês apenas. Erro: {e}")
        return {}, {}

def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def find_drug_matches(drug_input: str, drug_names_df: pd.DataFrame, realistic_drugs_df: pd.DataFrame, 
                     use_semantic: bool = False, drug_model=None, threshold: float = 0.6) -> List[str]:
    if use_semantic and drug_model is not None:
        semantic_results = semantic_drug_search(drug_input, drug_names_df, realistic_drugs_df, 
                                              drug_model, threshold=threshold, top_k=10)
        return [drug_name for drug_name, score in semantic_results]
    else:
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

def translate_symptom_to_english(symptom: str, pt_to_en_dict: Dict[str, str], model) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Tenta traduzir um sintoma em português para inglês.
    Retorna a melhor tradução encontrada e possíveis alternativas.
    """
    normalized_symptom = normalize_text(symptom)
    
    # Busca exata no dicionário
    if normalized_symptom in pt_to_en_dict:
        return pt_to_en_dict[normalized_symptom], [(pt_to_en_dict[normalized_symptom], 1.0)]
    
    # Busca parcial no dicionário
    partial_matches = []
    for pt_effect, en_effect in pt_to_en_dict.items():
        if normalized_symptom in pt_effect or pt_effect in normalized_symptom:
            partial_matches.append((en_effect, 0.9))
    
    if partial_matches:
        return partial_matches[0][0], partial_matches
    
    # Busca semântica se não houver correspondência direta
    if pt_to_en_dict:
        symptom_embedding = model.encode(normalized_symptom, convert_to_tensor=True)
        semantic_matches = []
        
        for pt_effect, en_effect in list(pt_to_en_dict.items())[:1000]:  # Limitar para performance
            effect_embedding = model.encode(normalize_text(pt_effect), convert_to_tensor=True)
            similarity = util.cos_sim(symptom_embedding, effect_embedding).item()
            if similarity > 0.5:
                semantic_matches.append((en_effect, similarity))
        
        if semantic_matches:
            semantic_matches.sort(key=lambda x: x[1], reverse=True)
            return semantic_matches[0][0], semantic_matches[:5]
    
    # Se não encontrar tradução, retorna o original
    return symptom, []

def check_side_effect_similarity(user_effect: str, known_effects: List[str], model, pt_to_en_dict: Dict[str, str] = None, en_to_pt_dict: Dict[str, str] = None) -> Tuple[List[Tuple[str, float]], str, List[Tuple[str, float]]]:
    """
    Verifica similaridade de efeitos colaterais, suportando entrada em português.
    Retorna: (similaridades, efeito_traduzido, possíveis_traduções)
    """
    if not known_effects:
        return [], user_effect, []
    
    # Tenta traduzir sintoma em português para inglês
    translated_effect = user_effect
    possible_translations = []
    
    if pt_to_en_dict:
        translated_effect, possible_translations = translate_symptom_to_english(user_effect, pt_to_en_dict, model)
    
    user_embedding = model.encode(normalize_text(translated_effect), convert_to_tensor=True)
    similarities = []
    
    for effect in known_effects:
        if effect and str(effect) != 'nan':
            effect_embedding = model.encode(normalize_text(str(effect)), convert_to_tensor=True)
            similarity = util.cos_sim(user_embedding, effect_embedding).item()
            similarities.append((effect, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True), translated_effect, possible_translations

def generate_ai_response(drug_name: str, user_effect: str, matches: List[Tuple[str, float]], drug_info: Dict, 
                        translated_effect: str = None, possible_translations: List[Tuple[str, float]] = None,
                        en_to_pt_dict: Dict[str, str] = None) -> str:
    """Gera resposta AI com suporte a traduções"""
    
    translation_info = ""
    if translated_effect and translated_effect.lower() != user_effect.lower():
        translation_info = f"\n**Sintoma informado:** {user_effect}\n**Tradução detectada:** {translated_effect}\n"
        if possible_translations and len(possible_translations) > 1:
            translation_info += f"**Traduções alternativas consideradas:** {', '.join([t[0] for t in possible_translations[:3]])}\n"
    
    if not matches:
        return f"""
        **Análise:** Não encontrei o efeito '{user_effect}' como um efeito colateral documentado para {drug_name} em nossa base de dados.
        {translation_info}
        **Recomendação:** Isso não significa que o efeito não possa estar relacionado ao medicamento. Reações individuais podem variar. 
        Recomendo consultar um médico ou farmacêutico para uma avaliação mais detalhada.
        
        **⚠️ Importante:** Este sistema é apenas informativo e não substitui orientação médica profissional.
        """
    
    best_match = matches[0]
    similarity_score = best_match[1]
    best_match_pt = en_to_pt_dict.get(best_match[0].lower(), best_match[0]) if en_to_pt_dict else best_match[0]
    
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
    {translation_info}
    **Efeito mais similar:** {best_match_pt} (Similaridade: {similarity_score:.2f})
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
drug_model = load_drug_model()
drug_names_df, realistic_drugs_df, sider_data_df = load_drug_data()
pt_to_en_dict, en_to_pt_dict = load_translations()

if drug_names_df is None:
    st.error("Erro ao carregar dados. Verifique se os arquivos CSV estão no diretório 'data/'.")
    st.stop()

st.title("💊 SynMed - Verificador de Efeitos Colaterais")
st.markdown("### Verifique se um sintoma pode ser efeito colateral de um medicamento")

with st.expander("⚙️ Configurações de Busca", expanded=False):
    use_semantic_search = st.checkbox(
        "Usar busca semântica para nomes de medicamentos", 
        value=True,
        help="Busca por similaridade semântica usando AI ao invés de correspondência exata"
    )
    
    if use_semantic_search:
        similarity_threshold = st.slider(
            "Limite de similaridade", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6, 
            step=0.1,
            help="Menor valor = mais resultados, Maior valor = resultados mais precisos"
        )
    else:
        similarity_threshold = 0.6

col1, col2 = st.columns(2)

with col1:
    drug_input = st.text_input(
        "Nome do medicamento:",
        placeholder="Ex: Paracetamol, Ibuprofeno, Amoxicilina"
    )

with col2:
    effect_input = st.text_input(
        "Efeito/sintoma observado (em português ou inglês):",
        placeholder="Ex: dor de cabeça, náusea, tontura, enjoo"
    )

if st.button("🔍 Verificar Efeito Colateral", type="primary"):
    if drug_input.strip() and effect_input.strip():
        with st.spinner("Analisando..."):
            drug_matches = find_drug_matches(
                drug_input, 
                drug_names_df, 
                realistic_drugs_df,
                use_semantic=use_semantic_search,
                drug_model=drug_model,
                threshold=similarity_threshold
            )
            
            if not drug_matches:
                search_type = "semântica" if use_semantic_search else "exata"
                st.warning(f"Medicamento '{drug_input}' não encontrado na base de dados usando busca {search_type}.")
                if use_semantic_search:
                    st.info("💡 Dica: Tente diminuir o limite de similaridade nas configurações ou desabilitar a busca semântica.")
            else:
                selected_drug = drug_matches[0]
                
                if len(drug_matches) > 1:
                    search_type = "semânticos" if use_semantic_search else "similares"
                    st.info(f"Encontrados múltiplos medicamentos {search_type}. Usando: {selected_drug}")
                    with st.expander("Ver todas as correspondências"):
                        for match in drug_matches:
                            st.write(f"- {match}")
                
                known_effects = get_side_effects_for_drug(selected_drug, realistic_drugs_df, sider_data_df, drug_names_df)
                
                similarities, translated_effect, possible_translations = check_side_effect_similarity(
                    effect_input, known_effects, sentence_model, pt_to_en_dict, en_to_pt_dict
                )
                
                drug_info = {}
                drug_row = realistic_drugs_df[realistic_drugs_df['drug_name'].str.contains(selected_drug, case=False, na=False)]
                if not drug_row.empty:
                    drug_info = drug_row.iloc[0].to_dict()
                
                ai_response = generate_ai_response(
                    selected_drug, effect_input, similarities, drug_info,
                    translated_effect, possible_translations, en_to_pt_dict
                )
                
                st.markdown("---")
                st.markdown("## Análise AI")
                st.markdown(ai_response)
                
                if similarities:
                    st.markdown("### Efeitos Colaterais Similares Conhecidos")
                    for i, (effect, score) in enumerate(similarities[:5]):
                        confidence_color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                        effect_pt = en_to_pt_dict.get(effect.lower(), effect) if en_to_pt_dict else effect
                        st.write(f"{confidence_color} **{effect_pt}** ({effect}) - Similaridade: {score:.3f}")
                
                if known_effects:
                    with st.expander(f"Ver todos os efeitos colaterais conhecidos de {selected_drug}"):
                        effects_list = []
                        for effect in sorted(set(known_effects)):
                            if effect and str(effect) != 'nan':
                                effect_pt = en_to_pt_dict.get(effect.lower(), effect) if en_to_pt_dict else effect
                                if effect_pt != effect:
                                    effects_list.append(f"• {effect_pt} ({effect})")
                                else:
                                    effects_list.append(f"• {effect}")
                        for effect_text in effects_list:
                            st.write(effect_text)
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
