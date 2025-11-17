import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Previsor ES - Tempo Real", layout="wide")

# --- FUNÇÃO PARA BUSCAR DADOS REAIS (OPEN-METEO) ---
@st.cache_data(ttl=3600) # Cache dura 1 hora para não sobrecarregar
def carregar_dados_reais():
    # Coordenadas de Vitória, Espírito Santo
    lat = "-20.3196"
    lon = "-40.3384"
    
    # API: Busca dados diários de temperatura máxima desde 2022 até hoje
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2022-01-01&end_date={date.today()}&daily=temperature_2m_max&timezone=America%2FSao_Paulo"
    
    response = requests.get(url)
    dados = response.json()
    
    # Transformar JSON em DataFrame Pandas
    df = pd.DataFrame({
        'data': dados['daily']['time'],
        'temp_max': dados['daily']['temperature_2m_max']
    })
    
    # Limpeza e Configuração
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    df.dropna(inplace=True) # Remove dias vazios se houver falha na API
    
    # Feature Engineering (Criar Lags)
    df['temp_ontem'] = df['temp_max'].shift(1)
    df['temp_anteontem'] = df['temp_max'].shift(2)
    df.dropna(inplace=True)
    
    return df

# --- CARREGAR E TREINAR ---
try:
    df = carregar_dados_reais()

    # Treinar Modelo
    X = df[['temp_ontem', 'temp_anteontem']]
    y = df['temp_max']
    
    # Usamos todos os dados históricos para treinar
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Pegar os dados mais recentes (reais) para exibir
    ultimos_dados = df.iloc[-1]
    temp_hoje_real = ultimos_dados['temp_max']
    temp_ontem_real = ultimos_dados['temp_ontem']

except Exception as e:
    st.error(f"Erro ao conectar com a API de clima: {e}")
    st.stop()

# --- INTERFACE DO SITE ---
st.title("🌤️ Previsão do Tempo - Espírito Santo (IA)")
st.markdown(f"**Dados Reais de Vitória/ES** | Última atualização: {date.today().strftime('%d/%m/%Y')}")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔮 Previsão para Amanhã")
    st.write("O modelo usa as temperaturas reais dos últimos 2 dias:")
    
    # Mostra os valores reais capturados da API
    st.info(f"🌡️ Máxima Hoje (Registrada): **{temp_hoje_real}°C**")
    st.info(f"🌡️ Máxima Ontem (Registrada): **{temp_ontem_real}°C**")
    
    # Faz a previsão automática
    previsao = modelo.predict([[temp_hoje_real, temp_ontem_real]])[0]
    
    st.success(f"🎯 Previsão da IA para Amanhã: **{previsao:.1f}°C**")
    
    st.markdown("---")
    st.caption("*Fonte de dados: Open-Meteo API (Historical Weather)*")

with col2:
    st.subheader("📊 Histórico Real (2024-2025)")
    
    # Filtrar para mostrar apenas os últimos 90 dias no gráfico para ficar bonito
    df_grafico = df.tail(90)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_grafico.index, df_grafico['temp_max'], color='orange', label='Temperatura Máxima')
    ax.set_title("Variação de Temperatura nos últimos 3 meses (Vitória-ES)")
    ax.set_ylabel("°C")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
