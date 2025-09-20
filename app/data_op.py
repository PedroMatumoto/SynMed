import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
import pickle
import os
import hashlib

def load_drug_search_model():
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")

def get_cache_key(drug_names: List[str]) -> str:
    combined_names = ''.join(sorted(drug_names))
    return hashlib.md5(combined_names.encode()).hexdigest()

def load_drug_embeddings_cache(cache_key: str) -> Dict:
    cache_file = f"cache/drug_embeddings_{cache_key}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_drug_embeddings_cache(cache_key: str, embeddings_data: Dict):
    os.makedirs("cache", exist_ok=True)
    cache_file = f"cache/drug_embeddings_{cache_key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_data, f)

def create_drug_embeddings(drug_names: List[str], model) -> Tuple[np.ndarray, Dict]:
    cache_key = get_cache_key(drug_names)
    
    cached_data = load_drug_embeddings_cache(cache_key)
    if cached_data is not None:
        return cached_data['embeddings'], cached_data['name_to_index']
    
    clean_names = [name.strip().lower() for name in drug_names if name and str(name) != 'nan']
    embeddings = model.encode(clean_names, convert_to_tensor=False, show_progress_bar=True)
    
    name_to_index = {name: idx for idx, name in enumerate(clean_names)}
    
    embeddings_data = {
        'embeddings': embeddings,
        'name_to_index': name_to_index,
        'drug_names': clean_names
    }
    save_drug_embeddings_cache(cache_key, embeddings_data)
    
    return embeddings, name_to_index

def semantic_drug_search(query: str, drug_names_df: pd.DataFrame, realistic_drugs_df: pd.DataFrame, 
                        model, threshold: float = 0.6, top_k: int = 10) -> List[Tuple[str, float]]:
    all_drugs = []
    drug_sources = {} 
    
    for _, row in drug_names_df.iterrows():
        drug_name = str(row['drug_name']).strip()
        if drug_name and drug_name != 'nan':
            all_drugs.append(drug_name)
            drug_sources[drug_name] = 'official'
    
    for _, row in realistic_drugs_df.iterrows():
        drug_name = str(row['drug_name']).strip()
        if drug_name and drug_name != 'nan' and drug_name not in drug_sources:
            all_drugs.append(drug_name)
            drug_sources[drug_name] = 'realistic'
    
    if not all_drugs:
        return []
    
    drug_embeddings, name_to_index = create_drug_embeddings(all_drugs, model)
    
    query_embedding = model.encode([query.strip().lower()], convert_to_tensor=False)
    
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
            results.append((original_name, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def treat_realistic_drugs(df: pd.DataFrame) -> pd.DataFrame:
    df['side_effects'] = df['side_effects'].fillna('').apply(lambda x: ', '.join([se.strip() for se in x.split(',') if se.strip() != '']))
    df = df[df['side_effects'] != '']
    df['drug_name'] = df['drug_name'].str.replace(r'\d+', '', regex=True).str.strip()
    df['side_effects'] = df['side_effects'].str.split(', ')
    df = df.explode('side_effects')
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    realistic_drugs = pd.read_csv('data/realistic_drug_labels_side_effects.csv')
    realistic_drugs = treat_realistic_drugs(realistic_drugs)
    print(realistic_drugs)