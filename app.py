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

# --- 2. ESTILIZAÇÃO (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #000428, #004e92);
        color: white;
    }
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
    h1, h2, h3, p, label, .stDateInput label { color: white !important; }
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
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
        
        # Lags para treinamento
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

# Treinamento do Modelo
X = df[['temp_ontem', 'temp_anteontem']]
y = df['temp_max']
modelo = LinearRegression().fit(X, y)

# Preparar dados atuais
ultimos = df.iloc[-1]
hoje_real = ultimos['temp_max']
ontem_real = ultimos['temp_ontem']

# --- 4. LÓGICA DE PREVISÃO FUTURA (RECURSIVA) ---
def prever_dias(qtd_dias):
    previsoes = []
    datas = []
    
    # Começamos com os dados conhecidos de hoje
    input_atual = [[hoje_real, ontem_real]]
    data_atual = date.today()
    
    for i in range(qtd_dias):
        # Prever o próximo dia
        pred = modelo.predict(input_atual)[0]
        
        data_futura = data_atual + timedelta(days=i+1)
        datas.append(data_futura)
        previsoes.append(pred)
        
        # Atualizar o input para o próximo passo (Recursão)
        # O que era "hoje" vira "ontem", e a previsão vira "hoje"
        novo_ontem = input_atual[0][0] # O antigo hoje
        novo_hoje = pred               # A nova previsão
        input_atual = [[novo_hoje, novo_ontem]]
        
    return datas, previsoes

# --- 5. INTERFACE VISUAL ---

st.title("🌤️ Clima ES Futuro")
st.markdown("<p style='text-align: center; color: #cccccc;'>Previsão Inteligente Recursiva</p>", unsafe_allow_html=True)
st.divider()

# Métricas do Dia
col1, col2 = st.columns(2)
with col1:
    st.metric("📅 Data Hoje", f"{date.today().strftime('%d/%m')}")
with col2:
    st.metric("🌡️ Temp. Hoje", f"{hoje_real}°C")

st.markdown("### 🔮 Simulador de Futuro")
st.write("Escolha uma data para ver a previsão estimada pela IA:")

# Input de Data (Limitado a 7 dias para não perder precisão)
max_date = date.today() + timedelta(days=7)
data_escolhida = st.date_input(
    "Data da Previsão",
    value=date.today() + timedelta(days=1),
    min_value=date.today() + timedelta(days=1),
    max_value=max_date
)

# Calcular quantos dias faltam até a data escolhida
dias_para_frente = (data_escolhida - date.today()).days

if dias_para_frente > 0:
    datas_fut, temps_fut = prever_dias(dias_para_frente)
    valor_previsto = temps_fut[-1]
    
    # Exibir Resultado Gigante
    st.markdown(f"""
        <div style='background: rgba(0, 210, 255, 0.15); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid rgba(0, 210, 255, 0.3); margin-bottom: 20px;'>
            <h3 style='margin:0; color: #e0e0e0;'>Previsão para {data_escolhida.strftime('%d/%m')}</h3>
            <h1 style='margin:0; font-size: 50px; color: #00d2ff;'>{valor_previsto:.1f}°C</h1>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico de Linha do Tempo (Mostra o caminho até lá)
    st.markdown("#### 📈 Trajetória Prevista")
    
    # Combinar dados recentes + futuros para o gráfico
    df_passado = df.tail(5).reset_index()[['data', 'temp_max']]
    df_passado['tipo'] = 'Real'
    
    df_futuro = pd.DataFrame({'data': pd.to_datetime(datas_fut), 'temp_max': temps_fut})
    df_futuro['tipo'] = 'Previsão'
    
    # Criar gráfico com duas cores
    fig = go.Figure()
    
    # Linha do Passado (Branca)
    fig.add_trace(go.Scatter(
        x=df_passado['data'], y=df_passado['temp_max'],
        mode='lines+markers', name='Histórico',
        line=dict(color='white', width=2)
    ))
    
    # Linha do Futuro (Azul Neon/Pontilhada)
    fig.add_trace(go.Scatter(
        x=[df_passado.iloc[-1]['data']] + list(df_futuro['data']), # Conecta o último ponto
        y=[df_passado.iloc[-1]['temp_max']] + list(df_futuro['temp_max']),
        mode='lines+markers', name='Previsão IA',
        line=dict(color='#00d2ff', width=3, dash='dot')
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Selecione uma data futura.")

# Rodapé
st.caption("Nota: A precisão diminui conforme a data se afasta (efeito cascata).")
