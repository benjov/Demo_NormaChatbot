import os
import pandas as pd
import zipfile
from openai import OpenAI
import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine

# --- CONFIGURACIÓN Y DATOS ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
#openai_api_key = os.getenv("OPENAI_API_KEY")

MODEL_LIST = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o']

default_system_message = (
    "Eres un asistente experto en criterios jurídicos de COFECE. "
    "Sólo puedes utilizar la información proporcionada dentro de los ### para responder, "
    "ignorando cualquier otro contexto. Si la información solicitada no está contenida en los criterios "
    "proporcionados, indícalo explícitamente al usuario."
)

st.sidebar.image("images/Logo_Norma.png", use_container_width=True)
st.sidebar.markdown("Contact: imanol@DOMINIO.com.mx ricardo@DOMINIO.com.mx benjamin@analiticaboutique.com.mx")
model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)
with st.sidebar:
    st.markdown("### Opciones avanzadas")
    show_system_message = st.checkbox("Mostrar mensaje de system en el chat", value=False)
    custom_system_message = st.text_area(
        "Mensaje de system para el asistente:",
        value = default_system_message,
        height = 300
    )

st.title("💬 Chatbot")
st.caption("🚀 un chat que te ayuda para tus actividades diarias")

#csv_file_path = "criterios_cofece_tabla_embedding_filtered.csv"
#df_filtered = pd.read_csv(csv_file_path)
#
zip_file_path = "criterios_cofece_tabla_embedding_filtered.zip"
csv_file_name = "criterios_cofece_tabla_embedding_filtered.csv"
with zipfile.ZipFile(zip_file_path) as z:
    with z.open(csv_file_name) as f:
        df_filtered = pd.read_csv(f)
#
df_filtered['ada_embedding'] = df_filtered['ada_embedding'].apply(eval).apply(np.array)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    client = OpenAI(api_key=openai_api_key)
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# --- ESTADO DE SESIÓN ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": custom_system_message},
        {"role": "assistant", "content": "¿Sobre qué tema te gustaría saber?"}
    ]
if "criterios_json" not in st.session_state:
    st.session_state["criterios_json"] = None
if "criterios_table" not in st.session_state:
    st.session_state["criterios_table"] = None
if "ultima_busqueda" not in st.session_state:
    st.session_state["ultima_busqueda"] = None
if "show_criterios_table" not in st.session_state:
    st.session_state["show_criterios_table"] = False

# --- IMPRESIÓN DEL HISTORIAL DE MENSAJES ---
for msg in st.session_state.messages:
    if msg["role"] != "system" or show_system_message:
        st.chat_message(msg["role"]).write(msg["content"])

if st.session_state["show_criterios_table"] and st.session_state["criterios_table"] is not None:
    st.write("Estos son los 10 criterios jurídicos más relevantes encontrados en tu última búsqueda:")
    st.table(st.session_state["criterios_table"].reset_index(drop=True))

# --- BOTÓN OPCIONAL PARA NUEVA BÚSQUEDA ---
st.markdown("---")
nueva_busqueda_btn = st.button("🔎 Nueva búsqueda de criterios")

# --- CAPTURA DEL CHAT ---
if prompt := st.chat_input():
    recalcular = False

    # 1. Si es la primera vez
    if st.session_state["criterios_json"] is None:
        recalcular = True

    # 2. Si el usuario hace clic en el botón
    if nueva_busqueda_btn:
        recalcular = True
        # El prompt actual se usa como nueva búsqueda

    # 3. Si el usuario usa /buscar
    if prompt.strip().lower().startswith("/buscar"):
        recalcular = True
        prompt = prompt.strip()[7:].strip()  # Quita el /buscar del texto

    # Si hay que recalcular los criterios
    if recalcular:
        Prompt_Embedding = np.array(get_embedding(prompt))
        def calcular_coseno(embedding):
            return cosine(embedding, Prompt_Embedding)
        df_filtered['cosine_distance'] = df_filtered['ada_embedding'].apply(calcular_coseno)
        df_sorted = df_filtered.sort_values('cosine_distance', ascending=True)
        df_top10 = df_sorted[['criterio', 'contenido']].head(10)
        st.session_state["criterios_table"] = df_top10
        json_output = df_top10.to_json(orient='records', indent=2, force_ascii=False)
        st.session_state["criterios_json"] = json_output
        st.session_state["ultima_busqueda"] = prompt.strip()
        st.session_state["show_criterios_table"] = True
    else:
        json_output = st.session_state["criterios_json"]
        st.session_state["show_criterios_table"] = False

    # Guarda el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Construye los mensajes para el modelo (siempre usa los criterios previos)
    model_messages = (
        st.session_state["messages"][:-1] + [
            {
                "role": "user",
                "content": prompt + "\n###" + json_output + "###"
            }
        ]
    )

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model_option,
        messages=model_messages,
        temperature=0
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
