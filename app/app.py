import json
import textwrap
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF

warnings.filterwarnings("ignore")

MODEL_PATH = Path("models/modelo_cardiaco_pipeline.pkl")
CONFIG_PATHS = [
    Path("models/config_modelo.json"),
    Path("config_modelo.json"),
]

st.set_page_config(
    page_title="CardioIA | Apoio à Avaliação Cardíaca",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# FUNÇÃO PARA GERAR RELATÓRIO PDF
def gerar_relatorio_pdf(dados_paciente, probabilidade, predicao, limiar):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cabeçalho / Título
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(10, 31, 51) # Navy
    pdf.cell(0, 10, "CardioIA - Relatorio de Avaliacao Clinica", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(92, 112, 137)
    pdf.cell(0, 5, "Suporte e Apoio a Decision Medica com Machine Learning", ln=True, align="C")
    pdf.ln(8)
    
    # Linha divisória
    pdf.set_draw_color(220, 228, 236)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    # Seção 1: Parâmetros Clínicos
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(10, 31, 51)
    pdf.cell(0, 8, "1. Dados do Paciente e Parametros Clinicos", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(15, 35, 55)
    
    col_width = 90
    items = list(dados_paciente.items())
    for i in range(0, len(items), 2):
        k1, v1 = items[i]
        txt1 = f"- {k1}: {v1}"
        if i + 1 < len(items):
            k2, v2 = items[i+1]
            txt2 = f"- {k2}: {v2}"
            pdf.cell(col_width, 6, txt1)
            pdf.cell(col_width, 6, txt2, ln=True)
        else:
            pdf.cell(col_width, 6, txt1, ln=True)
            
    pdf.ln(8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    # Seção 2: Resultado
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(10, 31, 51)
    pdf.cell(0, 8, "2. Resultado da Triagem (Machine Learning)", ln=True)
    pdf.ln(2)
    
    status_texto = "Maior Atencao (Classe Positiva)" if predicao == 1 else "Menor Atencao (Classe Negativa)"
    pct = probability * 100 if probability is not None else 0.0
    
    pdf.set_font("Helvetica", "B", 11)
    if predicao == 1:
        pdf.set_text_color(200, 30, 67) # Coral
    else:
        pdf.set_text_color(0, 168, 143) # Cyan Dark
        
    pdf.cell(0, 7, f"Classificacao do Modelo: {status_texto}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(15, 35, 55)
    pdf.cell(0, 6, f"Probabilidade Estimada: {pct:.1f}%", ln=True)
    pdf.cell(0, 6, f"Limiar da Decision: {limiar:.0%}", ln=True)
    
    pdf.ln(10)
    
    # Nota de Isenção / Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(122, 90, 10)
    pdf.set_fill_color(255, 248, 234)
    pdf.multi_cell(0, 5, "AVISO IMPORTANTE: O CardioIA e uma ferramenta academica de apoio. Nao realiza diagnostico definitivo, nao prescreve tratamento e nao substitui a avaliacao e conduta de um profissional medico habilitado.", border=1, fill=True)
    
    return bytes(pdf.output())


# TEMA VISUAL
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

        :root {
            --navy: #0A1F33;
            --navy-2: #123049;
            --cyan: #00E6C3;
            --cyan-dark: #00A88F;
            --coral: #FF4D6D;
            --coral-dark: #C81E43;
            --amber: #FFB020;
            --ink: #0F2337;
            --muted: #5C7089;
            --line: #DCE4EC;
            --surface: #FFFFFF;
            --page: #EFF3F8;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", Arial, sans-serif;
        }

        h1, h2, h3, .panel-title, .hero-title, .result-title {
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
        }

        .mono, .prob-value, .status-badge, .section-label,
        .threshold-row, div[data-testid="stMetricValue"] {
            font-family: "JetBrains Mono", monospace !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% -6%, rgba(0, 230, 195, 0.10), transparent 30%),
                radial-gradient(circle at 100% 0%, rgba(255, 77, 109, 0.08), transparent 26%),
                var(--page);
        }

        .block-container {
            max-width: 1260px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        /* ============ HERO ============ */
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 26px;
            padding: 38px 38px 30px 38px;
            margin-bottom: 24px;
            color: white;
            background: linear-gradient(128deg, #071726 0%, #0A1F33 40%, #0E3A4A 78%, #0B4A45 100%);
            box-shadow: 0 22px 55px rgba(6, 20, 33, 0.30);
            border: 1px solid rgba(255,255,255,0.06);
        }

        .hero-ecg {
            position: absolute;
            left: 0;
            right: 0;
            bottom: -6px;
            width: 100%;
            height: 78px;
            opacity: 0.55;
        }

        .hero-ecg path {
            fill: none;
            stroke: var(--cyan);
            stroke-width: 2.2;
            stroke-linecap: round;
            stroke-linejoin: round;
            stroke-dasharray: 900;
            stroke-dashoffset: 900;
            animation: draw-ecg 3.6s ease-in-out infinite;
            filter: drop-shadow(0 0 6px rgba(0, 230, 195, 0.55));
        }

        @keyframes draw-ecg {
            0%   { stroke-dashoffset: 900; }
            55%  { stroke-dashoffset: 0; }
            100% { stroke-dashoffset: -900; }
        }

        @media (prefers-reduced-motion: reduce) {
            .hero-ecg path { animation: none; stroke-dashoffset: 0; }
        }

        .hero-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            flex-wrap: wrap;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(0, 230, 195, 0.12);
            border: 1px solid rgba(0, 230, 195, 0.35);
            color: #7CF5E1;
        }

        .hero-kicker .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 0 4px rgba(0,230,195,0.18);
        }

        .hero-title {
            font-size: 3.1rem;
            line-height: 1.0;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin: 16px 0 0 0;
        }

        .hero-title span {
            color: var(--cyan);
        }

        .hero-subtitle {
            max-width: 720px;
            margin-top: 14px;
            margin-bottom: 0;
            font-size: 1.02rem;
            line-height: 1.6;
            color: rgba(255,255,255,0.82);
            position: relative;
            z-index: 2;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
            position: relative;
            z-index: 2;
        }

        .hero-chip {
            display: inline-flex;
            align-items: baseline;
            gap: 6px;
            padding: 8px 13px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.14);
            backdrop-filter: blur(2px);
        }

        .hero-chip b {
            font-family: "JetBrains Mono", monospace;
            color: var(--cyan);
            font-size: 0.86rem;
        }

        /* ============ PAINÉIS ============ */
        .panel {
            background: rgba(255,255,255,0.97);
            border: 1px solid var(--line);
            border-left: 4px solid var(--cyan-dark);
            border-radius: 18px;
            padding: 20px 22px;
            box-shadow: 0 10px 26px rgba(15, 35, 55, 0.06);
            margin-bottom: 14px;
        }

        .panel-title {
            color: var(--navy);
            font-size: 1.32rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 4px 0;
        }

        .panel-subtitle {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }

        .section-label {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--cyan-dark);
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-top: 6px;
            margin-bottom: 10px;
        }

        .section-label::before {
            content: "";
            width: 16px;
            height: 2px;
            background: var(--cyan-dark);
            display: inline-block;
        }

        /* ============ RESULTADO ============ */
        .result-shell {
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px 22px;
            box-shadow: 0 10px 26px rgba(15, 35, 55, 0.06);
        }

        .result-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 12px;
            border-radius: 10px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .status-badge::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }

        .status-high { background: #FFE4E9; color: #C81E43; }
        .status-high::before { background: #C81E43; box-shadow: 0 0 0 3px rgba(200,30,67,0.18); }

        .status-low { background: #DBFAF2; color: #00806C; }
        .status-low::before { background: #00806C; box-shadow: 0 0 0 3px rgba(0,128,108,0.18); }

        .status-neutral { background: #E9EEF4; color: #52677A; }
        .status-neutral::before { background: #8CA0B3; }

        .result-main {
            position: relative;
            border-radius: 16px;
            padding: 20px;
            margin-top: 8px;
            overflow: hidden;
        }

        .result-main.high {
            background: linear-gradient(135deg, #FFF2F4, #FFFAFA);
            border: 1px solid #FFC9D3;
        }

        .result-main.low {
            background: linear-gradient(135deg, #EAFBF6, #F7FDFB);
            border: 1px solid #B9EEDD;
        }

        .result-main.neutral {
            background: linear-gradient(135deg, #F7FAFC, #FBFCFD);
            border: 1px solid #E2EAF0;
        }

        .result-title {
            color: var(--navy);
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
        }

        .result-text {
            color: #506276;
            font-size: 0.92rem;
            line-height: 1.5;
            margin: 0;
        }

        .prob-wrap {
            margin-top: 16px;
            padding: 16px 17px;
            border-radius: 14px;
            background: var(--navy);
            border: 1px solid var(--navy-2);
        }

        .prob-top {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 10px;
            margin-bottom: 10px;
        }

        .prob-label {
            color: #9FB4C7;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .prob-value {
            color: var(--cyan);
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .bar-bg {
            height: 10px;
            width: 100%;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            overflow: hidden;
        }

        .bar-fill-high {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #FF8FA3, var(--coral));
            box-shadow: 0 0 10px rgba(255,77,109,0.6);
        }

        .bar-fill-low {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #4DEFD4, var(--cyan-dark));
            box-shadow: 0 0 10px rgba(0,230,195,0.5);
        }

        .threshold-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-top: 10px;
            color: #7C90A3;
            font-size: 0.72rem;
        }

        .clinical-note {
            margin-top: 14px;
            padding: 13px 15px;
            border-left: 4px solid var(--amber);
            background: #FFF8EA;
            border-radius: 10px;
            color: #7A5A0A;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .empty-card {
            text-align: left;
            border-radius: 16px;
            padding: 20px;
            background: linear-gradient(135deg, #F6FAFD, #FFFFFF);
            border: 1px dashed #B9CBDA;
        }

        .empty-card strong {
            color: var(--navy);
            font-family: "Space Grotesk", sans-serif;
        }

        /* ============ COMPONENTES STREAMLIT ============ */
        .stNumberInput label,
        .stSelectbox label {
            color: #1F3A54 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 11px !important;
            border-color: var(--line) !important;
        }

        div[data-testid="stNumberInput"] input:focus,
        div[data-baseweb="select"]:focus-within > div {
            border-color: var(--cyan-dark) !important;
            box-shadow: 0 0 0 3px rgba(0,168,143,0.16) !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            min-height: 50px;
            border-radius: 12px;
            border: none;
            font-weight: 700;
            font-size: 0.98rem;
            letter-spacing: 0.01em;
            background: linear-gradient(90deg, var(--cyan-dark), #067A6A);
            color: white;
            box-shadow: 0 10px 22px rgba(0,168,143,0.28);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(90deg, #00CBB0, var(--cyan-dark));
            color: white;
            border: none;
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(0,168,143,0.36);
        }

        div[data-testid="stDownloadButton"] > button {
            margin-top: 15px;
            width: 100%;
            border-radius: 12px;
            background: #0A1F33;
            color: #00E6C3;
            border: 1px solid #123049;
            font-weight: 600;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(255,255,255,0.88);
        }

        div[data-testid="stMetric"] {
            background: var(--navy);
            border: 1px solid var(--navy-2);
            border-radius: 13px;
            padding: 13px 15px;
        }

        div[data-testid="stMetricLabel"] {
            color: #9FB4C7 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--cyan) !important;
        }

        hr {
            border-color: #E1E9F0 !important;
        }

        #MainMenu, footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# CARREGAMENTO
def find_config_path():
    for path in CONFIG_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "config_modelo.json não encontrado em models/ nem na pasta principal."
    )


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    config_path = find_config_path()

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, config


try:
    model, config = load_assets()
except Exception as error:
    st.error(
        "Não foi possível carregar o modelo e sua configuração. "
        "Verifique a pasta models e o arquivo config_modelo.json."
    )
    with st.expander("Detalhes técnicos"):
        st.code(str(error))
    st.stop()


THRESHOLD = float(config.get("threshold", 0.5))
FEATURES_ESPERADAS = config.get("features_esperadas", [])



# FUNÇÕES
def build_input_dataframe(
    age,
    sex,
    trestbps,
    chol,
    fbs,
    thalch,
    exang,
    oldpeak,
    cp_type,
    restecg,
    slope,
    thal,
):
    input_data = {
        "id": 0,
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Sim" else 0,
        "thalch": thalch,
        "exang": 1 if exang == "Sim" else 0,
        "oldpeak": oldpeak,
        "sex_Male": 1 if sex == "Masculino" else 0,
        "cp_atypical angina": 1 if cp_type == "Angina atípica" else 0,
        "cp_non-anginal": 1 if cp_type == "Dor não anginosa" else 0,
        "cp_typical angina": 1 if cp_type == "Angina típica" else 0,
        "restecg_normal": 1 if restecg == "Normal" else 0,
        "restecg_st-t abnormality": (
            1 if restecg == "Anormalidade de ST-T" else 0
        ),
        "slope_flat": 1 if slope == "Plano" else 0,
        "slope_upsloping": 1 if slope == "Ascendente" else 0,
        "thal_normal": 1 if thal == "Normal" else 0,
        "thal_reversable defect": 1 if thal == "Defeito reversível" else 0,
    }

    df_input = pd.DataFrame([input_data])

    if FEATURES_ESPERADAS:
        missing = [col for col in FEATURES_ESPERADAS if col not in df_input.columns]

        if missing:
            raise ValueError(
                "A interface não gerou todas as variáveis esperadas pelo modelo: "
                + ", ".join(missing)
            )

        df_input = df_input[FEATURES_ESPERADAS]

    return df_input


def validate_model_features(df_input):
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is not None:
        if list(model_features) != list(df_input.columns):
            raise ValueError(
                "As colunas enviadas pela interface não coincidem com as colunas "
                "gravadas no modelo treinado."
            )


def predict_with_threshold(df_input):
    if hasattr(model, "predict_proba"):
        probability_positive = float(model.predict_proba(df_input)[0][1])
        prediction = int(probability_positive >= THRESHOLD)
        return prediction, probability_positive

    prediction = int(model.predict(df_input)[0])
    return prediction, None


def probability_html(probability, prediction):
    pct = max(0.0, min(100.0, probability * 100))
    fill_class = "bar-fill-high" if prediction == 1 else "bar-fill-low"

    return (
        f'<div class="prob-wrap">'
        f'<div class="prob-top">'
        f'<span class="prob-label">PROBABILIDADE · CLASSE POSITIVA</span>'
        f'<span class="prob-value">{pct:.1f}%</span>'
        f'</div>'
        f'<div class="bar-bg">'
        f'<div class="{fill_class}" style="width:{pct:.1f}%"></div>'
        f'</div>'
        f'<div class="threshold-row">'
        f'<span>LIMIAR: {THRESHOLD:.0%}</span>'
        f'<span>CALCULADO AUTOMATICAMENTE</span>'
        f'</div>'
        f'</div>'
    )



# CABEÇALHO
st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">Cardio<span>IA</span></h1>
        <p class="hero-subtitle">
            Uma interface para apoiar a análise inicial de dados cardíacos durante o
            preenchimento do prontuário, usando Machine Learning como suporte
            complementar à avaliação profissional.
        </p>
        <div class="hero-chips">
            <span class="hero-chip"><b>Machine&nbsp;Learning</b> clínico</span>
            <span class="hero-chip">Base <b>UCI&nbsp;Heart&nbsp;Disease</b></span>
            <span class="hero-chip">Apoio ao <b>prontuário</b></span>
        </div>
        <svg class="hero-ecg" viewBox="0 0 1200 80" preserveAspectRatio="none">
            <path d="M0,40 L180,40 L210,40 L225,12 L245,68 L262,40 L300,40 L520,40 L545,40 L560,14 L580,66 L597,40 L640,40 L860,40 L885,40 L900,12 L920,68 L937,40 L980,40 L1200,40" />
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)


# CONTEÚDO PRINCIPAL
col_form, col_result = st.columns([1.45, 0.85], gap="large")

with col_form:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Dados do paciente</div>
            <p class="panel-subtitle">
                Informe os dados clínicos usados pelo modelo. Os campos foram
                organizados em dois grupos para deixar o preenchimento mais rápido.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("cardio_form", clear_on_submit=False):
        st.markdown('<div class="section-label">Informações gerais</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Idade (anos)",
                min_value=18,
                max_value=120,
                value=None,
                step=1,
                placeholder="Ex.: 55",
            )

            sex = st.selectbox(
                "Sexo biológico",
                options=["Feminino", "Masculino"],
                index=None,
                placeholder="Selecione",
            )

            trestbps = st.number_input(
                "Pressão arterial em repouso (mmHg)",
                min_value=50,
                max_value=250,
                value=None,
                step=1,
                placeholder="Ex.: 130",
            )

        with c2:
            chol = st.number_input(
                "Colesterol sérico (mg/dL)",
                min_value=50,
                max_value=700,
                value=None,
                step=1,
                placeholder="Ex.: 240",
            )

            fbs = st.selectbox(
                "Glicemia em jejum acima de 120 mg/dL",
                options=["Não", "Sim"],
                index=None,
                placeholder="Selecione",
            )

            thalch = st.number_input(
                "Frequência cardíaca máxima alcançada",
                min_value=40,
                max_value=230,
                value=None,
                step=1,
                placeholder="Ex.: 150",
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Avaliação cardíaca</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            cp_type = st.selectbox(
                "Tipo de dor torácica",
                options=[
                    "Assintomático / outro",
                    "Angina típica",
                    "Angina atípica",
                    "Dor não anginosa",
                ],
                index=None,
                placeholder="Selecione",
            )

            restecg = st.selectbox(
                "Eletrocardiograma em repouso",
                options=[
                    "Normal",
                    "Anormalidade de ST-T",
                    "Hipertrofia ventricular esquerda",
                ],
                index=None,
                placeholder="Selecione",
            )

            exang = st.selectbox(
                "Angina induzida por exercício",
                options=["Não", "Sim"],
                index=None,
                placeholder="Selecione",
            )

        with c4:
            oldpeak = st.number_input(
                "Depressão do segmento ST (oldpeak)",
                min_value=-5.0,
                max_value=10.0,
                value=None,
                step=0.1,
                placeholder="Ex.: 1,0",
            )

            slope = st.selectbox(
                "Inclinação do segmento ST",
                options=["Ascendente", "Plano", "Descendente"],
                index=None,
                placeholder="Selecione",
            )

            thal = st.selectbox(
                "Resultado do exame thal",
                options=["Normal", "Defeito fixo", "Defeito reversível"],
                index=None,
                placeholder="Selecione",
                help="Campo mantido conforme a variável utilizada no treinamento.",
            )

        submitted = st.form_submit_button(
            "Avaliar dados",
            type="primary",
            use_container_width=True,
        )


with col_result:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Resultado</div>
            <p class="panel-subtitle">
                A classificação aparece aqui assim que os dados forem processados.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not submitted:
        st.markdown(
            textwrap.dedent(
                """\
                <div class="result-shell">
                    <div class="result-header-row">
                        <span class="status-badge status-neutral">Aguardando dados</span>
                    </div>

                    <div class="empty-card">
                        <strong>Nenhuma avaliação realizada.</strong>
                        <div style="margin-top:6px; color:#66788A; font-size:0.9rem; line-height:1.45;">
                            Preencha os dados ao lado e clique em <b>Avaliar dados</b>.
                            O resultado será exibido neste espaço.
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    else:
        values = {
            "Idade": age,
            "Sexo biológico": sex,
            "Pressão arterial": trestbps,
            "Colesterol": chol,
            "Glicemia em jejum": fbs,
            "Frequência cardíaca máxima": thalch,
            "Tipo de dor torácica": cp_type,
            "ECG em repouso": restecg,
            "Angina por exercício": exang,
            "Oldpeak": oldpeak,
            "Inclinação ST": slope,
            "Thal": thal,
        }

        missing_fields = [name for name, value in values.items() if value is None]

        if missing_fields:
            st.warning(
                "Preencha todos os campos antes de realizar a avaliação. "
                "Campos pendentes: " + ", ".join(missing_fields)
            )
        else:
            try:
                df_input = build_input_dataframe(
                    age=age,
                    sex=sex,
                    trestbps=trestbps,
                    chol=chol,
                    fbs=fbs,
                    thalch=thalch,
                    exang=exang,
                    oldpeak=oldpeak,
                    cp_type=cp_type,
                    restecg=restecg,
                    slope=slope,
                    thal=thal,
                )

                validate_model_features(df_input)
                prediction, probability = predict_with_threshold(df_input)

                if prediction == 1:
                    result_block = (
                        '<div class="result-shell">'
                        '<div class="result-header-row">'
                        '<span class="status-badge status-high">Maior atenção</span>'
                        '</div>'
                        '<div class="result-main high">'
                        '<div class="result-title">Classe positiva do modelo</div>'
                        '<p class="result-text">'
                        'O modelo encontrou um padrão associado à classe positiva. '
                        'O resultado deve ser revisado pelo profissional responsável.'
                        '</p>'
                        '</div>'
                    )
                else:
                    result_block = (
                        '<div class="result-shell">'
                        '<div class="result-header-row">'
                        '<span class="status-badge status-low">Menor atenção</span>'
                        '</div>'
                        '<div class="result-main low">'
                        '<div class="result-title">Classe negativa do modelo</div>'
                        '<p class="result-text">'
                        'O modelo classificou o registro na classe negativa. '
                        'Isso não exclui doença e não substitui avaliação clínica.'
                        '</p>'
                        '</div>'
                    )

                if probability is not None:
                    result_block += probability_html(probability, prediction)

                result_block += (
                    '<div class="clinical-note">'
                    '<b>Importante:</b> o CardioIA é uma ferramenta acadêmica de apoio. '
                    'Não realiza diagnóstico definitivo, não prescreve tratamento '
                    'e não deve ser usado isoladamente para definir conduta.'
                    '</div>'
                    '</div>'
                )

                st.markdown(result_block, unsafe_allow_html=True)

                # ============ BOTÃO PARA GERAR E BAIXAR O RELATÓRIO PDF ============
                pdf_bytes = gerar_relatorio_pdf(
                    dados_paciente=values,
                    probabilidade=probability,
                    predicao=prediction,
                    limiar=THRESHOLD
                )

                st.download_button(
                    label="📄 Baixar Relatório em PDF",
                    data=pdf_bytes,
                    file_name="relatorio_cardioia.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                with st.expander("Ver dados enviados ao modelo"):
                    st.dataframe(
                        df_input,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as error:
                st.error(
                    "Não foi possível concluir a avaliação. "
                    "Verifique a compatibilidade entre a interface e o modelo treinado."
                )
                with st.expander("Detalhes técnicos"):
                    st.code(str(error))


# INFORMAÇÕES DO MODELO
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Informações do modelo"):
    st.write(
        "Esta versão utiliza o limiar definido no arquivo de configuração "
        "e mantém as mesmas variáveis esperadas pelo modelo treinado."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Limiar de classificação", f"{THRESHOLD:.0%}")

    if "recall_no_threshold" in config:
        m2.metric(
            "Recall documentado",
            f"{float(config['recall_no_threshold']) * 100:.1f}%",
        )
    else:
        m2.metric("Recall documentado", "Não informado")

    if "precision_no_threshold" in config:
        m3.metric(
            "Precisão documentada",
            f"{float(config['precision_no_threshold']) * 100:.1f}%",
        )
    else:
        m3.metric("Precisão documentada", "Não informado")

    st.caption(
        "Nota técnica: o modelo atual ainda espera a coluna 'id'. "
        "Ela foi mantida apenas por compatibilidade e deve ser removida em um futuro retreinamento."
    )
