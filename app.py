import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Previsão ES",
    page_icon="🌤️",
    layout="centered"
)

# --- 2. ESTILIZAÇÃO AVANÇADA (CSS) ---
st.markdown("""
    <style>
    /* === 1. Mudança Global de Cores (Tira o Vermelho) === */
    :root {
        --primary-color: #00d2ff; /* Azul Neon como cor de foco */
    }
    
    /* Fundo Principal */
    .stApp {
        background: linear-gradient(to bottom right, #000428, #004e92);
        color: white;
    }
    
    /* === 2. Estilização do Calendário e Inputs === */
    /* Borda do input quando selecionado */
    input:focus, div[data-baseweb="input"]:focus-within {
        border-color: #00d2ff !important;
        box-shadow: 0 0 0 1px #00d2ff !important;
    }
    /* Cor do calendário interno */
    div[data-baseweb="calendar"] {
        background-color: #001529 !important;
    }
    
    /* === 3. Cards de Vidro === */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricLabel"] { color: #e0e0e0; font-size: 14px; }
    div[data-testid="stMetricValue"] { color: #ffffff; font-size: 28px; }
    
    /* Textos Gerais */
    h1, h2, h3, p, label, .stDateInput label { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DADOS E IA ---
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
    st.error("Erro de conexão. Tente recarregar.")
    st.stop()

# Treinamento
X = df[['temp_ontem', 'temp_anteontem']]
y = df['temp_max']
modelo = LinearRegression().fit(X, y)

# Preparar dados atuais
ultimos = df.iloc[-1]
hoje_real = ultimos['temp_max']
ontem_real = ultimos['temp_ontem']

# --- 4. LÓGICA DE PREVISÃO FUTURA ---
def prever_dias(qtd_dias):
    previsoes = []
    datas = []
    input_atual = [[hoje_real, ontem_real]]
    data_atual = date.today()
    
    for i in range(qtd_dias):
        pred = modelo.predict(input_atual)[0]
        data_futura = data_atual + timedelta(days=i+1)
        datas.append(data_futura)
        previsoes.append(pred)
        
        novo_ontem = input_atual[0][0]
        novo_hoje = pred
        input_atual = [[novo_hoje, novo_ontem]]
        
    return datas, previsoes

# --- 5. INTERFACE VISUAL ---

st.title("🌤️ Clima ES Futuro")
st.markdown("<p style='text-align: center; color: #cccccc;'>Previsão Inteligente Recursiva</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.metric("📅 Data Hoje", f"{date.today().strftime('%d/%m')}")
with col2:
    st.metric("🌡️ Temp. Hoje", f"{hoje_real}°C")

st.markdown("### 🔮 Simulador de Futuro")
st.write("Escolha uma data para ver a previsão:")

# Calendário
max_date = date.today() + timedelta(days=7)
data_escolhida = st.date_input(
    "Data da Previsão",
    value=date.today() + timedelta(days=1),
    min_value=date.today() + timedelta(days=1),
    max_value=max_date
)

dias_para_frente = (data_escolhida - date.today()).days

if dias_para_frente > 0:
    datas_fut, temps_fut = prever_dias(dias_para_frente)
    valor_previsto = temps_fut[-1]
    
    # Card de Resultado Grande
    st.markdown(f"""
        <div style='background: rgba(0, 210, 255, 0.15); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid rgba(0, 210, 255, 0.3); margin-bottom: 20px;'>
            <h3 style='margin:0; color: #e0e0e0;'>Previsão para {data_escolhida.strftime('%d/%m')}</h3>
            <h1 style='margin:0; font-size: 50px; color: #00d2ff; text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);'>{valor_previsto:.1f}°C</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📈 Trajetória (Histórico + Previsão)")
    
    # --- CORREÇÃO DO GRÁFICO ---
    # Aumentei de tail(5) para tail(45) para mostrar muito mais histórico
    df_passado = df.tail(45).reset_index()[['data', 'temp_max']]
    df_passado['tipo'] = 'Real'
    
    df_futuro = pd.DataFrame({'data': pd.to_datetime(datas_fut), 'temp_max': temps_fut})
    df_futuro['tipo'] = 'Previsão'
    
    fig = go.Figure()
    
    # Linha Histórica
    fig.add_trace(go.Scatter(
        x=df_passado['data'], y=df_passado['temp_max'],
        mode='lines', name='Histórico Recente',
        line=dict(color='white', width=2),
        hovertemplate='%{x|%d/%m}: %{y:.1f}°C<extra></extra>' # Tooltip bonito
    ))
    
    # Linha Futura (Conectada)
    fig.add_trace(go.Scatter(
        x=[df_passado.iloc[-1]['data']] + list(df_futuro['data']),
        y=[df_passado.iloc[-1]['temp_max']] + list(df_futuro['temp_max']),
        mode='lines+markers', name='Previsão IA',
        line=dict(color='#00d2ff', width=3, dash='dot'),
        marker=dict(size=6, color='#00d2ff'),
        hovertemplate='%{x|%d/%m}: %{y:.1f}°C<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode="x unified", # Faz uma linha vertical ao passar o mouse
        xaxis=dict(
            showgrid=False, 
            title='',
            fixedrange=False # Permite zoom e arrastar
        ),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Selecione uma data futura.")
