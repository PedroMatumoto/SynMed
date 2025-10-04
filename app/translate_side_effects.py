import pandas as pd
from deep_translator import GoogleTranslator
import time
from pathlib import Path

def extract_unique_side_effects(file_path):
    """Extract unique side effects from meddra_all_se.csv"""
    print("Lendo o arquivo CSV...")
    df = pd.read_csv(file_path, sep=';', header=None)
    
    # A última coluna contém os efeitos colaterais
    # Colunas: CID1, CID2, Code1, Type, Code2, Side Effect Name
    df.columns = ['CID1', 'CID2', 'Code1', 'Type', 'Code2', 'Side_Effect']
    
    # Extrair valores únicos da coluna de efeitos colaterais
    unique_effects = df['Side_Effect'].unique()
    
    print(f"Total de efeitos colaterais únicos encontrados: {len(unique_effects)}")
    return sorted(unique_effects)

def translate_side_effects(side_effects, batch_size=50):
    """Translate side effects from English to Portuguese"""
    translator = GoogleTranslator(source='en', target='pt')
    
    translations = {}
    total = len(side_effects)
    
    print(f"\nIniciando tradução de {total} efeitos colaterais...")
    
    for i, effect in enumerate(side_effects, 1):
        try:
            # Traduzir o efeito
            translated = translator.translate(effect)
            translations[effect] = translated
            
            # Mostrar progresso
            if i % 10 == 0 or i == total:
                print(f"Progresso: {i}/{total} ({i/total*100:.1f}%)")
            
            # Pequena pausa para evitar rate limiting
            if i % batch_size == 0:
                time.sleep(1)
                
        except Exception as e:
            print(f"Erro ao traduzir '{effect}': {e}")
            translations[effect] = effect  # Manter original em caso de erro
            time.sleep(2)
    
    return translations

def save_translations(translations, output_path):
    """Save translations to CSV file"""
    df = pd.DataFrame(list(translations.items()), 
                     columns=['Original', 'Traduzido'])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nTraduções salvas em: {output_path}")

def main():
    # Caminhos dos arquivos
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'meddra_all_se.csv'
    output_file = base_dir / 'data' / 'side_effects_translated.csv'
    
    # Extrair efeitos únicos
    unique_effects = extract_unique_side_effects(input_file)
    
    # Mostrar alguns exemplos
    print("\nPrimeiros 10 efeitos colaterais:")
    for effect in unique_effects[:10]:
        print(f"  - {effect}")
    
    # Confirmar tradução
    print("\nDeseja continuar com a tradução? (s/n)")
    response = input().strip().lower()
    
    if response != 's':
        print("Tradução cancelada.")
        return
    
    # Traduzir
    translations = translate_side_effects(unique_effects)
    
    # Salvar resultados
    save_translations(translations, output_file)
    
    # Mostrar alguns exemplos de traduções
    print("\nExemplos de traduções:")
    for i, (original, translated) in enumerate(list(translations.items())[:10], 1):
        print(f"{i}. {original} -> {translated}")
    
    print(f"\nTotal de traduções realizadas: {len(translations)}")

if __name__ == "__main__":
    main()
