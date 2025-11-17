import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Previsor Universitário", layout="wide")

# --- FUNÇÃO DE DADOS (CACHEADA) ---
@st.cache_resource
def treinar_modelo():
    # Gera dados sintéticos
    np.random.seed(42)
    dias = pd.date_range(start='2023-01-01', periods=365, freq='D')
    temperatura = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, 365)) + np.random.normal(0, 2, 365)
    
    df = pd.DataFrame({'data': dias, 'temp': temperatura}).set_index('data')
    df['temp_ontem'] = df['temp'].shift(1)
    df['temp_anteontem'] = df['temp'].shift(2)
    df.dropna(inplace=True)
    
    X = df[['temp_ontem', 'temp_anteontem']]
    y = df['temp']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    return modelo, df.iloc[-30:]

modelo, df_recente = treinar_modelo()

# --- INTERFACE ---
st.title("🌤️ Previsor de Temperatura")
st.markdown("Sistema desenvolvido para a disciplina de Análise Temporal.")

col1, col2 = st.columns([1, 2])

with col1:
    st.info("Insira os dados abaixo:")
    with st.form("meu_form"):
        t1 = st.number_input("Temperatura Ontem", value=25.0)
        t2 = st.number_input("Temperatura Anteontem", value=24.0)
        enviar = st.form_submit_button("Calcular Previsão")
        
    if enviar:
        pred = modelo.predict([[t1, t2]])[0]
        st.success(f"Previsão para Amanhã: **{pred:.2f}°C**")

with col2:
    st.subheader("Histórico Recente")
    st.line_chart(df_recente['temp'])