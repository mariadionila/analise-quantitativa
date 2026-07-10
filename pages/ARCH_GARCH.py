import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from arch import arch_model

from ui import apply_theme, hero, style_axis


st.set_page_config(page_title="ARCH/GARCH", layout="wide")
apply_theme()

hero(
    "ARCH/GARCH",
    "Modele a volatilidade condicional dos ativos e acompanhe a persistência do risco ao longo do tempo.",
    "Volatilidade",
)

st.sidebar.header("Configurações")

lista_ativos = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
    "WEGE3.SA",
    "BBAS3.SA",
    "MGLU3.SA",
    "SUZB3.SA",
    "B3SA3.SA",
    "RENT3.SA",
    "LREN3.SA",
]

ativos = st.sidebar.multiselect(
    "Selecione os ativos",
    lista_ativos,
    default=["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
)

data_inicio = st.sidebar.date_input("Data inicial", pd.to_datetime("2020-01-01"))
data_fim = st.sidebar.date_input("Data final", pd.to_datetime("today"))

if st.sidebar.button("Executar Análise"):
    resultados_garch = []

    for ativo in ativos:
        try:
            st.header(ativo)

            dados_ativo = yf.download(ativo, start=data_inicio, end=data_fim, progress=False)

            if dados_ativo.empty:
                st.warning(f"Sem dados para {ativo}")
                continue

            retorno = dados_ativo["Close"].pct_change().dropna()

            if len(retorno) < 30:
                st.warning(f"Poucos dados para GARCH em {ativo}")
                continue

            modelo_garch = arch_model(retorno * 100, vol="Garch", p=1, q=1)
            resultado = modelo_garch.fit(disp="off")
            volatilidade = resultado.conditional_volatility
            previsao = resultado.forecast(horizon=5)
            volatilidade_prevista = previsao.variance.iloc[-1].mean()

            resultados_garch.append(
                {
                    "Ativo": ativo,
                    "Omega": round(resultado.params["omega"], 6),
                    "Alpha": round(resultado.params["alpha[1]"], 4),
                    "Beta": round(resultado.params["beta[1]"], 4),
                    "Volatilidade Prevista": round(volatilidade_prevista, 4),
                }
            )

            st.subheader("Evolução da Volatilidade Condicional")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(volatilidade, label="Volatilidade", color="#22d3ee")
            ax.set_title(f"GARCH(1,1) - {ativo}")
            ax.set_xlabel("Tempo")
            ax.set_ylabel("Volatilidade")
            ax.legend()
            style_axis(fig, ax)
            st.pyplot(fig)
        except Exception as erro:
            st.error(f"Erro GARCH ({ativo}): {erro}")

    if resultados_garch:
        df_garch = pd.DataFrame(resultados_garch)

        st.header("Resultado Final GARCH")
        st.dataframe(df_garch, use_container_width=True)

        st.subheader("Comparação de Volatilidade")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(df_garch["Ativo"], df_garch["Volatilidade Prevista"], color="#a3e635")
        ax2.set_title("Volatilidade Prevista")
        ax2.set_ylabel("Volatilidade")
        style_axis(fig2, ax2)
        st.pyplot(fig2)
    else:
        st.warning("Nenhum ativo válido para análise.")
