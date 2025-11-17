import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import qrcode
from PIL import Image
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import io

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Previsão ES",
    page_icon="🌤️",
    layout="centered"
)

# --- 2. ESTILIZAÇÃO ---
st.markdown("""
    <style>
    :root { --primary-color: #00d2ff; }
    .stApp { background: linear-gradient(to bottom right, #000428, #004e92); color: white; }
    input:focus, div[data-baseweb="input"]:focus-within { border-color: #00d2ff !important; box-shadow: 0 0 0 1px #00d2ff !important; }
    div[data-baseweb="calendar"] { background-color: #001529 !important; }
    div[data-testid="stMetric"] { background-color: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 15px; text-align: center; backdrop-filter: blur(10px); }
    div[data-testid="stMetricLabel"] { color: #e0e0e0; font-size: 14px; }
    div[data-testid="stMetricValue"] { color: #ffffff; font-size: 28px; }
    h1, h2, h3, p, label, .stDateInput label { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. BARRA LATERAL COM QR CODE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=50)
    st.markdown("### Acesse no Celular")
    
    # Pegar a URL atual (Em produção, use a URL real do seu app)
    # COLOQUE AQUI O SEU LINK DO STREAMLIT CLOUD
    url_do_app = "https://previsao-tempo-faculdade-dupnxup5yddv24jmzswppv.streamlit.app" 
    
    # Gerar QR Code
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(url_do_app)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")
    
    # Converter para imagem exibível
    buffer = io.BytesIO()
    img_qr.save(buffer, format="PNG")
    st.image(buffer, caption="Escaneie para testar agora!", use_container_width=True)
    
    st.info("Projeto de Análise Temporal\nAluno: [Seu Nome]")

# --- 4. DADOS E IA ---
@st.cache_data(ttl=3600)
def carregar_dados():
    lat, lon = "-20.3196", "-40.3384"
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date={date.today()}&daily=temperature_2m_max&timezone=America%2FSao_Paulo"
    try:
        response = requests.get(url)
        dados = response.json()
        df = pd.DataFrame({'data': dados['daily']['time'], 'temp_max': dados['daily']['temperature_2m_max']})
        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        df.dropna(inplace=True)
        df['temp_ontem'] = df['temp_max'].shift(1)
        df['temp_anteontem'] = df['temp_max'].shift(2)
        df.dropna(inplace=True)
        return df
    except: return None

df = carregar_dados()
if df is None: st.stop()

X = df[['temp_ontem', 'temp_anteontem']]
y = df['temp_max']
modelo = LinearRegression().fit(X, y)
ultimos = df.iloc[-1]
hoje_real, ontem_real = ultimos['temp_max'], ultimos['temp_ontem']

def prever_dias(qtd_dias):
    previsoes, datas = [], []
    input_atual = [[hoje_real, ontem_real]]
    data_atual = date.today()
    for i in range(qtd_dias):
        pred = modelo.predict(input_atual)[0]
        data_futura = data_atual + timedelta(days=i+1)
        datas.append(data_futura)
        previsoes.append(pred)
        input_atual = [[pred, input_atual[0][0]]]
    return datas, previsoes

# --- 5. INTERFACE ---
st.title("🌤️ Clima ES Futuro")
st.divider()

col1, col2 = st.columns(2)
col1.metric("📅 Data Hoje", f"{date.today().strftime('%d/%m')}")
col2.metric("🌡️ Temp. Hoje", f"{hoje_real}°C")

st.markdown("### 🔮 Simulador de Futuro")
max_date = date.today() + timedelta(days=7)
data_escolhida = st.date_input("Data da Previsão", value=date.today()+timedelta(days=1), min_value=date.today()+timedelta(days=1), max_value=max_date)

dias = (data_escolhida - date.today()).days
if dias > 0:
    datas_fut, temps_fut = prever_dias(dias)
    st.markdown(f"<div style='background: rgba(0, 210, 255, 0.15); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid rgba(0, 210, 255, 0.3); margin-bottom: 20px;'><h3 style='margin:0; color: #e0e0e0;'>Previsão para {data_escolhida.strftime('%d/%m')}</h3><h1 style='margin:0; font-size: 50px; color: #00d2ff; text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);'>{temps_fut[-1]:.1f}°C</h1></div>", unsafe_allow_html=True)
    
    df_passado = df.tail(45).reset_index()[['data', 'temp_max']]
    df_futuro = pd.DataFrame({'data': pd.to_datetime(datas_fut), 'temp_max': temps_fut})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_passado['data'], y=df_passado['temp_max'], mode='lines', name='Histórico', line=dict(color='white', width=2), hovertemplate='%{x|%d/%m}: %{y:.1f}°C<extra></extra>'))
    fig.add_trace(go.Scatter(x=[df_passado.iloc[-1]['data']] + list(df_futuro['data']), y=[df_passado.iloc[-1]['temp_max']] + list(df_futuro['temp_max']), mode='lines+markers', name='Previsão IA', line=dict(color='#00d2ff', width=3, dash='dot'), marker=dict(size=6, color='#00d2ff'), hovertemplate='%{x|%d/%m}: %{y:.1f}°C<extra></extra>'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), hovermode="x unified", xaxis=dict(showgrid=False, title=''), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


