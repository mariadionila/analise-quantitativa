import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from pandas_datareader import data as web
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Modelo Fama-French",
    page_icon="📈",
    layout="wide"
)

st.title("Modelo Fama-French (3 Fatores)")
st.markdown(
    "Analise o comportamento dos ativos em relação aos fatores de mercado "
    "(**Mercado**, **SMB** e **HML**)."
)

# Sidebar
st.sidebar.header("Configurações")

ativos = st.sidebar.multiselect(
    "Selecione os ativos",
    [
        "PETR4.SA",
        "VALE3.SA",
        "ITUB4.SA",
        "BBDC4.SA",
        "ABEV3.SA",
        "WEGE3.SA"
    ],
    default=["PETR4.SA", "VALE3.SA"]
)

inicio = st.sidebar.date_input(
    "Data Inicial",
    pd.to_datetime("2023-01-01")
)

fim = st.sidebar.date_input(
    "Data Final",
    pd.to_datetime("today")
)

executar = st.sidebar.button("Executar Análise")

if executar:

    with st.spinner("Baixando fatores Fama-French..."):

        fatores = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench"
        )[0]

        fatores.index = fatores.index.to_timestamp()
        fatores = fatores / 100

    resultados = []

    barra = st.progress(0)

    for i, ativo in enumerate(ativos):

        try:

            dados = yf.download(
                ativo,
                start=inicio,
                end=fim,
                progress=False,
                auto_adjust=True
            )

            close = dados["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            retorno = close.pct_change().dropna()

            retorno.index = pd.to_datetime(retorno.index).normalize()
            fatores.index = pd.to_datetime(fatores.index).normalize()

            df = pd.concat(
                [
                    retorno.rename("Retorno"),
                    fatores
                ],
                axis=1,
                join="inner"
            ).dropna()

            X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
            y = df["Retorno"] - df["RF"]

            modelo = sm.OLS(y, X).fit()

            resultados.append({
                "Ativo": ativo,
                "Alpha": modelo.params["const"],
                "Beta Mercado": modelo.params["Mkt-RF"],
                "SMB": modelo.params["SMB"],
                "HML": modelo.params["HML"],
                "R²": modelo.rsquared
            })

        except Exception as e:
            st.error(f"{ativo}: {e}")

        barra.progress((i + 1) / len(ativos))

    if resultados:

        df_resultados = pd.DataFrame(resultados)

        st.success("Análise concluída!")

        st.divider()

        st.subheader("Indicadores")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Ativos",
            len(df_resultados)
        )

        c2.metric(
            "Maior Beta",
            f"{df_resultados['Beta Mercado'].max():.2f}"
        )

        c3.metric(
            "Maior Alpha",
            f"{df_resultados['Alpha'].max():.4f}"
        )

        c4.metric(
            "Maior R²",
            f"{df_resultados['R²'].max():.2%}"
        )

        st.divider()

        st.subheader("Resultados")

        st.dataframe(
            df_resultados.style
            .background_gradient(
                cmap="Blues",
                subset=["Beta Mercado", "SMB", "HML", "R²"]
            )
            .format({
                "Alpha": "{:.4f}",
                "Beta Mercado": "{:.2f}",
                "SMB": "{:.2f}",
                "HML": "{:.2f}",
                "R²": "{:.2%}"
            }),
            use_container_width=True
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(7,4))

            ax.bar(
                df_resultados["Ativo"],
                df_resultados["Beta Mercado"]
            )

            ax.set_title("Beta de Mercado")
            ax.grid(axis="y", alpha=0.3)

            st.pyplot(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(7,4))

            ax.bar(
                df_resultados["Ativo"],
                df_resultados["Alpha"]
            )

            ax.set_title("Alpha")
            ax.grid(axis="y", alpha=0.3)

            st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:

            fig, ax = plt.subplots(figsize=(7,4))

            ax.bar(
                df_resultados["Ativo"],
                df_resultados["SMB"]
            )

            ax.set_title("Fator SMB")
            ax.grid(axis="y", alpha=0.3)

            st.pyplot(fig)

        with col4:

            fig, ax = plt.subplots(figsize=(7,4))

            ax.bar(
                df_resultados["Ativo"],
                df_resultados["HML"]
            )

            ax.set_title("Fator HML")
            ax.grid(axis="y", alpha=0.3)

            st.pyplot(fig)

        st.divider()

        fig, ax = plt.subplots(figsize=(10,4))

        ax.bar(
            df_resultados["Ativo"],
            df_resultados["R²"]
        )

        ax.set_title("Qualidade do Ajuste (R²)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)

        st.divider()

        st.subheader("Interpretação")

        for _, linha in df_resultados.iterrows():

            st.info(
                f"""
### {linha['Ativo']}

**Alpha:** {linha['Alpha']:.4f}

**Beta Mercado:** {linha['Beta Mercado']:.2f}

**SMB:** {linha['SMB']:.2f}

**HML:** {linha['HML']:.2f}

**R²:** {linha['R²']:.2%}
"""
            )