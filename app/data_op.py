import pandas as pd

def treat_realistic_drugs(df: pd.DataFrame) -> pd.DataFrame:
    df['side_effects'] = df['side_effects'].fillna('').apply(lambda x: ', '.join([se.strip() for se in x.split(',') if se.strip() != '']))
    df = df[df['side_effects'] != '']
    # filtrar numeros da coluna drug_name
    df['drug_name'] = df['drug_name'].str.replace(r'\d+', '', regex=True).str.strip()
    # caso tenha mais de um sintoma, quebrar em linhas separadas (sintomas aparecem como "side_effect1, side_effect2")
    df['side_effects'] = df['side_effects'].str.split(', ')
    df = df.explode('side_effects')
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Example usage
    realistic_drugs = pd.read_csv('data/realistic_drug_labels_side_effects.csv')
    realistic_drugs = treat_realistic_drugs(realistic_drugs)
    print(realistic_drugs)