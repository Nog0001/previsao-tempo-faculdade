import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px # Nova biblioteca de gráficos
from sklearn.linear_model import LinearRegression
from datetime import date

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Previsor ES - Tempo Real", layout="wide")

# --- FUNÇÃO DE DADOS ---
@st.cache_data(ttl=3600)
def carregar_dados_reais():
    # Vitória - ES
    lat, lon = "-20.3196", "-40.3384"
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date={date.today()}&daily=temperature_2m_max&timezone=America%2FSao_Paulo"
    
    response = requests.get(url)
    dados = response.json()
    
    df = pd.DataFrame({
        'data': dados['daily']['time'],
        'temp_max': dados['daily']['temperature_2m_max']
    })
    
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    df.dropna(inplace=True)
    
    # Criar Lags para o modelo
    df['temp_ontem'] = df['temp_max'].shift(1)
    df['temp_anteontem'] = df['temp_max'].shift(2)
    df.dropna(inplace=True)
    
    return df

# --- LÓGICA PRINCIPAL ---
try:
    df = carregar_dados_reais()

    # Treino
    X = df[['temp_ontem', 'temp_anteontem']]
    y = df['temp_max']
    modelo = LinearRegression().fit(X, y)
    
    # Dados para exibição
    ultimos_dados = df.iloc[-1]
    temp_hoje = ultimos_dados['temp_max']
    temp_ontem = ultimos_dados['temp_ontem']
    
    # Previsão
    predicao = modelo.predict([[temp_hoje, temp_ontem]])[0]

except Exception as e:
    st.error("Erro ao carregar dados. Tente recarregar a página.")
    st.stop()

# --- INTERFACE ---
st.title("🌤️ Clima Inteligente: Vitória-ES")
st.markdown(f"Monitoramento em tempo real e previsão via Machine Learning.")

# Colunas
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.metric("Temperatura Hoje (Real)", f"{temp_hoje}°C")

with col_kpi2:
    # Delta mostra a diferença para ontem
    diferenca = temp_hoje - temp_ontem
    st.metric("Variação desde Ontem", f"{diferenca:.1f}°C", delta=f"{diferenca:.1f}°C")

with col_kpi3:
    st.metric("Previsão IA para Amanhã", f"{predicao:.1f}°C", delta_color="off")

st.divider()

# --- GRÁFICO INTERATIVO (PLOTLY) ---
st.subheader("📈 Evolução da Temperatura (Interativo)")
st.info("Passe o mouse sobre o gráfico para ver detalhes ou arraste para dar zoom.")

# Pegamos os últimos 6 meses para não ficar pesado, mas mantemos o histórico
df_grafico = df.tail(180).reset_index()

# Criando o gráfico bonito
fig = px.line(df_grafico, x='data', y='temp_max', 
              title='Histórico Recente (Vitória-ES)',
              labels={'data': 'Data', 'temp_max': 'Temperatura Máxima (°C)'},
              markers=True) # Adiciona bolinhas nos pontos

# Personalizando visual
fig.update_traces(line_color='#FF4B4B', line_width=3) # Cor vermelha padrão Streamlit
fig.update_layout(hovermode="x unified") # Mostra linha vertical ao passar mouse

# Exibir no Streamlit
st.plotly_chart(fig, use_container_width=True)
