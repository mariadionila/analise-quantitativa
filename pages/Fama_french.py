import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from pandas_datareader import data as web
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fama-French", layout="wide")
st.title("Modelo Fama-French (3 Fatores)")

ativos = st.multiselect(
    "Selecione os ativos",
    ["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA","WEGE3.SA"],
    default=["PETR4.SA","VALE3.SA"]
)

inicio = st.date_input("Data Inicial", pd.to_datetime("2023-01-01"))
fim = st.date_input("Data Final", pd.to_datetime("today"))

if st.button("Executar Análise"):

    st.info("Baixando fatores Fama-French...")

    fatores = web.DataReader(
        "F-F_Research_Data_Factors_daily",
        "famafrench"
    )[0]

    fatores.index = pd.to_datetime(fatores.index)
    fatores = fatores / 100

    resultados = []

    for ativo in ativos:
        try:
            dados = yf.download(
                ativo,
                start=inicio,
                end=fim,
                progress=False
            )

            retorno = dados["Close"].pct_change().dropna()
            retorno.name = "Retorno"

            df = pd.concat([retorno, fatores], axis=1, join="inner").dropna()

            X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
            y = df["Retorno"] - df["RF"]

            modelo = sm.OLS(y, X).fit()

            resultados.append({
                "Ativo": ativo,
                "Alpha": round(modelo.params["const"], 6),
                "Beta Mercado": round(modelo.params["Mkt-RF"], 4),
                "SMB": round(modelo.params["SMB"], 4),
                "HML": round(modelo.params["HML"], 4),
                "R²": round(modelo.rsquared, 4)
            })

        except Exception as e:
            st.error(f"{ativo}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)

        st.subheader("Resultados")
        st.dataframe(df_resultados, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(df_resultados["Ativo"], df_resultados["Beta Mercado"])
        ax.set_title("Beta de Mercado")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.bar(df_resultados["Ativo"], df_resultados["SMB"])
        ax2.set_title("Fator SMB")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10,5))
        ax3.bar(df_resultados["Ativo"], df_resultados["HML"])
        ax3.set_title("Fator HML")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
