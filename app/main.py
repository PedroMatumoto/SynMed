import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


@st.cache_resource
def load_model():
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")

@st.cache_resource
def load_llm():
    model_id = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    return tokenizer, model

tokenizer, model = load_llm()
sentence_model = load_model()

def extract_symptoms(text):
    prompt = f"Você é um extrator de sintomas. Extraia os sintomas do seguinte texto: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    texto = tokenizer.decode(output[0], skip_special_tokens=True)
    return texto.split("Sintomas:")[-1].strip()

data = {
    "Remédios": [
        "Paracetamol",
        "Ibuprofeno",
        "Dipirona",
        "Antihistamínicos"
    ],
    "Sintomas": [
        "dor no corpo",
        "tosse seca",
        "dor de cabeça intensa",
        "espirros"
    ]
}
df = pd.DataFrame(data)
print(df)

df["embedding"] = df["Sintomas"].apply(lambda x: sentence_model.encode(x, convert_to_tensor=True))

st.title("🔎 Diagnóstico por Similaridade Semântica")

chat = st.chat_message("human")
chat.subheader("Olá! Eu sou o SynMed, seu assistente de saúde.")
chat.write("Digite os sintomas que você está sentindo:")

sintomas_usuario = chat.text_area(
    label="Input", 
    label_visibility="hidden", 
    placeholder="Ex: dor de cabeça, febre, tosse seca"
)

if st.button("Buscar"):
    if sintomas_usuario.strip() != "":
        with st.spinner("Processando com gpt-oss-20b..."):
            sintomas_extraidos = extract_symptoms(sintomas_usuario)
        chat = st.chat_message("assistant")
        chat.subheader("Sintomas extraídos:")
        chat.write(sintomas_extraidos)
        query_emb = sentence_model.encode(sintomas_extraidos, convert_to_tensor=True)

        scores = [util.cos_sim(query_emb, emb).item() for emb in df["embedding"]]
        df["similaridade"] = scores

        resultados = df.sort_values("similaridade", ascending=False).head(3)

        chat.subheader("Resultados mais semelhantes:")
        for _, row in resultados.iterrows():
            chat.write(f"**Doença:** {row['Doenca']}")
            chat.write(f"**Sintomas cadastrados:** {row['Sintomas']}")
            chat.write(f"**Similaridade:** {row['similaridade']:.2f}")
            chat.markdown("---")

        with st.spinner("Consultando a LLM para recomendações..."):
            top_remedio = resultados.iloc[0]["Remédios"]
            prompt = f"Monte uma resposta amigável para o usuário, recomendando o remédio '{top_remedio}' com base nos sintomas '{sintomas_extraidos}'."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=100)
            resposta = tokenizer.decode(output[0], skip_special_tokens=True)
            chat = st.chat_message("assistant")
            chat.subheader("Resposta da LLM:")
            chat.write(resposta)
    else:
        st.warning("Por favor, digite os sintomas antes de buscar.")
