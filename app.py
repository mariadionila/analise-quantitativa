import streamlit as st

from ui import apply_theme, hero


st.set_page_config(
    page_title="Painel de Inteligencia Financeira",
    layout="wide",
)

apply_theme()

hero(
    "Painel de Inteligencia Financeira",
    "Ambiente para estudar ativos brasileiros com metricas de risco, fatores de retorno, volatilidade e modelos preditivos.",
    "Mercado brasileiro",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cobertura", "B3")
c2.metric("Etapas", "Risco + IA")
c3.metric("Modelos", "7")
c4.metric("Uso", "Decisao")

st.write("")

col1, col2 = st.columns([1.05, 1])

with col1:
    st.markdown(
        """
        <div class="card">
            <span class="badge">Primeiro bloco</span>
            <h3>Diagnostico de risco</h3>
            <p>CAPM, Fama-French e ARCH/GARCH ajudam a separar retorno esperado, exposicao ao mercado e volatilidade condicional.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <span class="badge">Segundo bloco</span>
            <h3>Predicao e teste historico</h3>
            <p>Modelos de machine learning, redes recorrentes e backtesting avaliam o comportamento futuro dos retornos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown(
        """
        <div class="card">
            <span class="badge">01</span>
            <h3>CAPM</h3>
            <p>Beta, alpha, retorno esperado e premio de risco para medir exposicao ao benchmark.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="card">
            <span class="badge">02</span>
            <h3>Fama-French</h3>
            <p>Analise multifatorial com mercado, tamanho (SMB) e valor (HML).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        """
        <div class="card">
            <span class="badge">03</span>
            <h3>Volatilidade</h3>
            <p>ARCH/GARCH para acompanhar instabilidade e persistencia de risco nos ativos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Navegue pelas paginas no menu lateral para executar cada modulo.")
