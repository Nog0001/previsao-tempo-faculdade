import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import date

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Previsão ES",
    page_icon="🌤️",
    layout="centered" # 'Centered' fica muito melhor em celulares do que 'Wide'
)

# --- 2. ESTILIZAÇÃO (CSS PERSONALIZADO) ---
# Aqui criamos o fundo degradê e os cards estilo "vidro"
st.markdown("""
    <style>
    /* Fundo Degradê (Céu) */
    .stApp {
        background: linear-gradient(to bottom right, #000428, #004e92);
        color: white;
    }
    
    /* Estilo dos Cards (Métricas) */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* Cor dos textos das métricas */
    div[data-testid="stMetricLabel"] {
        color: #e0e0e0;
        font-size: 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 28px;
    }
    
    /* Título Centralizado */
    h1 {
        text-align: center;
        color: white;
        text-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    
    /* Ajuste do Gráfico */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. DADOS E IA (Mesma lógica robusta) ---
@st.cache_data(ttl=3600)
def carregar_dados():
    # Vitória - ES
    lat, lon = "-20.3196", "-40.3384"
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date={date.today()}&daily=temperature_2m_max&timezone=America%2FSao_Paulo"
    
    try:
        response = requests.get(url)
        dados = response.json()
        df = pd.DataFrame({
            'data': dados['daily']['time'],
            'temp_max': dados['daily']['temperature_2m_max']
        })
        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        df.dropna(inplace=True)
        
        # Lags
        df['temp_ontem'] = df['temp_max'].shift(1)
        df['temp_anteontem'] = df['temp_max'].shift(2)
        df.dropna(inplace=True)
        return df
    except:
        return None

df = carregar_dados()

if df is None:
    st.error("Erro ao conectar na API. Verifique a internet.")
    st.stop()

# Treinamento
X = df[['temp_ontem', 'temp_anteontem']]
y = df['temp_max']
modelo = LinearRegression().fit(X, y)

# Dados Recentes
ultimos = df.iloc[-1]
hoje_real = ultimos['temp_max']
ontem_real = ultimos['temp_ontem']
previsao = modelo.predict([[hoje_real, ontem_real]])[0]

# --- 4. INTERFACE VISUAL ---

st.title("🌤️ Clima ES")
st.markdown("<p style='text-align: center; color: #cccccc;'>Previsão via Machine Learning com dados reais.</p>", unsafe_allow_html=True)

st.markdown("---")

# Layout de Colunas (No celular, o Streamlit empilha isso automaticamente)
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.metric("📅 Data", f"{date.today().strftime('%d/%m')}")
with col2:
    st.metric("📍 Local", "Vitória, ES")

with col3:
    st.metric("🌡️ Hoje (Real)", f"{hoje_real}°C")
with col4:
    # Destacando a previsão
    st.metric("🔮 Previsão Amanhã", f"{previsao:.1f}°C", delta_color="normal")

st.markdown("### 📈 Tendência Recente")

# Gráfico Otimizado para Dark Mode
df_grafico = df.tail(60).reset_index()
fig = px.area(df_grafico, x='data', y='temp_max', 
              title='', 
              labels={'data': '', 'temp_max': 'Temp (°C)'})

# Personalizando o Plotly para combinar com o fundo escuro
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', # Fundo transparente
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
)
fig.update_traces(line_color='#00d2ff', fill_color='rgba(0, 210, 255, 0.2)')

st.plotly_chart(fig, use_container_width=True)

# Botãozinho de recarregar camuflado
if st.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()
