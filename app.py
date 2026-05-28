import streamlit as st

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Análise Quantitativa",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

/* Fundo */
.stApp {
    background-color: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Texto sidebar */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Títulos */
h1, h2, h3 {
    color: white;
}

/* Texto */
p, div {
    color: #cbd5e1;
}

/* Cards */
.card {

    background: #1e293b;

    padding: 20px;

    border-radius: 15px;

    border: 1px solid #334155;

    transition: 0.3s;
}

/* Hover */
.card:hover {

    border: 1px solid #38bdf8;

    transform: scale(1.02);
}

/* Métricas */
[data-testid="metric-container"] {

    background-color: #1e293b;

    border: 1px solid #334155;

    padding: 15px;

    border-radius: 12px;
}

/* Botão */
.stButton>button {

    background-color: #38bdf8;

    color: black;

    border-radius: 10px;

    border: none;

    font-weight: bold;
}

.stButton>button:hover {

    background-color: #0ea5e9;

    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("📈 Análise Quantitativa")

st.caption(
    "Sistema de análise quantitativa"
)

# =====================================================
# KPIs
# =====================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Modelos", "3")
c2.metric("Ativos", "12")
c3.metric("Framework", "Streamlit")
c4.metric("Área", "Quant")

# =====================================================
# ESPAÇO
# =====================================================

st.write("")

# =====================================================
# CARDS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="card">

    ## 📊 CAPM

    Modelo de Risco e Retorno

    - Beta
    - Retorno Esperdo
    - Mercado de Risco

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="card">

    ## 📈 Fama-French

    Modelo Multifatorial

    - SMB
    - HML
    - Fatores de Risco

    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="card">

    ## 📉 ARCH/GARCH

    Modelos de Volatilidade

    - Variância Condicional
    - Dinâmica do Risco
    - Previsão

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.write("")
st.write("")

st.caption(
    "Quantitative Finance • Econometrics • Time Series"
)
