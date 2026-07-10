import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from pandas_datareader import data as web

from ui import apply_theme, hero, style_axis


st.set_page_config(page_title="Fama-French", layout="wide")
apply_theme()

hero(
    "Modelo Fama-French",
    "Analise a exposição dos retornos aos fatores de mercado, tamanho (SMB) e valor (HML).",
    "Modelo multifatorial",
)

ativos = st.multiselect(
    "Selecione os ativos",
    ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA"],
    default=["PETR4.SA", "VALE3.SA"],
)

inicio = st.date_input("Data inicial", pd.to_datetime("2023-01-01"))
fim = st.date_input("Data final", pd.to_datetime("today"))

if st.button("Executar Análise"):
    st.info("Baixando fatores Fama-French...")

    fatores = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
    fatores.index = pd.to_datetime(fatores.index)
    fatores = fatores / 100

    resultados = []

    for ativo in ativos:
        try:
            dados = yf.download(ativo, start=inicio, end=fim, progress=False)
            retorno = dados["Close"].pct_change().dropna()
            retorno.name = "Retorno"

            df = pd.concat([retorno, fatores], axis=1, join="inner").dropna()

            X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
            y = df["Retorno"] - df["RF"]
            modelo = sm.OLS(y, X).fit()

            resultados.append(
                {
                    "Ativo": ativo,
                    "Alpha": round(modelo.params["const"], 6),
                    "Beta Mercado": round(modelo.params["Mkt-RF"], 4),
                    "SMB": round(modelo.params["SMB"], 4),
                    "HML": round(modelo.params["HML"], 4),
                    "R²": round(modelo.rsquared, 4),
                }
            )
        except Exception as erro:
            st.error(f"{ativo}: {erro}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)

        st.subheader("Resultados")
        st.dataframe(df_resultados, use_container_width=True)

        charts = [
            ("Beta de Mercado", "Beta Mercado", "#22d3ee"),
            ("Fator SMB", "SMB", "#a3e635"),
            ("Fator HML", "HML", "#f472b6"),
        ]

        for titulo, coluna, cor in charts:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_resultados["Ativo"], df_resultados[coluna], color=cor)
            ax.set_title(titulo)
            style_axis(fig, ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
