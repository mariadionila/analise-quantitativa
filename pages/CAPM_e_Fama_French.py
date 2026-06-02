import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# configuração da página

st.set_page_config(
    page_title="Fase I - CAPM",
    layout="wide"
)

st.title("📈 Fase I - Filtro de Risco e Econometria")
st.markdown("## Modelo CAPM para Seleção de Ativos")

# parâmetros

st.sidebar.header("Parâmetros da Análise")

# lista de ativos disponíveis

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
    "JBSS3.SA",
    "RENT3.SA",
    "LREN3.SA"
]

# seleção dos ativos

ativos = st.sidebar.multiselect(
    "Selecione os ativos",
    lista_ativos,
    default=[
        "PETR4.SA",
        "VALE3.SA",
        "ITUB4.SA",
        "BBDC4.SA",
        "ABEV3.SA",
        "WEGE3.SA",
        "BBAS3.SA"
    ]
)

benchmark = "^BVSP"

data_inicio = st.sidebar.date_input(
    "Data Inicial",
    pd.to_datetime("2023-01-01")
)

data_fim = st.sidebar.date_input(
    "Data Final",
    pd.to_datetime("today")
)

taxa_livre = st.sidebar.number_input(
    "Taxa Livre de Risco (%)",
    value=10.75
) / 100

st.sidebar.markdown("### Ativos analisados")

for ativo in ativos:
    st.sidebar.write(ativo)

# botão

if st.sidebar.button("Executar Análise CAPM"):

    st.subheader("📥 Coletando Dados...")

    # download benchmark

    dados_benchmark = yf.download(
        benchmark,
        start=data_inicio,
        end=data_fim,
        progress=False
    )

    retorno_mercado = (
        dados_benchmark["Close"]
        .pct_change()
        .dropna()
    )

    # lista de resultados

    resultados = []

    # loop dos ativos

    for ativo in ativos:

        try:

            # download dos dados

            dados_ativo = yf.download(
                ativo,
                start=data_inicio,
                end=data_fim,
                progress=False
            )

            # retornos

            retorno_ativo = (
                dados_ativo["Close"]
                .pct_change()
                .dropna()
            )

            # dataframe conjunto

            df = pd.concat(
                [retorno_ativo, retorno_mercado],
                axis=1
            ).dropna()

            df.columns = [
                "Ativo",
                "Mercado"
            ]

            # regressão capm

            X = sm.add_constant(df["Mercado"])

            y = df["Ativo"]

            modelo = sm.OLS(y, X).fit()

            # extrair alpha e beta

            alpha = modelo.params["const"]

            beta = modelo.params["Mercado"]

            # retorno esperado

            retorno_medio_mercado = (
                df["Mercado"].mean() * 252
            )

            retorno_esperado = (
                taxa_livre +
                beta * (
                    retorno_medio_mercado - taxa_livre
                )
            )

            # prêmio de risco

            premio_risco = (
                retorno_esperado - taxa_livre
            )

            # volatilidade

            volatilidade = (
                df["Ativo"].std() * np.sqrt(252)
            )

            # r²

            r2 = modelo.rsquared

            # salvar resultados

            resultados.append({
                "Ativo": ativo,
                "Alpha": round(alpha, 6),
                "Beta": round(beta, 4),
                "Retorno Esperado (%)": round(retorno_esperado * 100, 2),
                "Prêmio de Risco (%)": round(premio_risco * 100, 2),
                "Volatilidade (%)": round(volatilidade * 100, 2),
                "R²": round(r2, 4)
            })

        except Exception as e:

            st.error(f"Erro ao processar {ativo}: {e}")

    # dataframe final

    df_resultados = pd.DataFrame(resultados)

    # filtro dos melhores

    df_resultados = df_resultados.sort_values(
        by="Prêmio de Risco (%)",
        ascending=False
    )

    # resultados

    st.subheader("Resultado Final do CAPM")

    st.dataframe(
        df_resultados,
        use_container_width=True
    )

    # métricas gerais

    st.subheader(" Estatísticas Gerais")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Maior Beta",
        df_resultados.loc[
            df_resultados["Beta"].idxmax(),
            "Ativo"
        ]
    )

    col2.metric(
        "Menor Beta",
        df_resultados.loc[
            df_resultados["Beta"].idxmin(),
            "Ativo"
        ]
    )

    col3.metric(
        "Maior Prêmio de Risco",
        df_resultados.loc[
            df_resultados["Prêmio de Risco (%)"].idxmax(),
            "Ativo"
        ]
    )

    # gráfico beta

    st.subheader(" Beta dos Ativos")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        df_resultados["Ativo"],
        df_resultados["Beta"]
    )

    ax1.set_title("Beta dos Ativos")

    ax1.set_ylabel("Beta")

    st.pyplot(fig1)

    # gráfico retorno esperado

    st.subheader(" Retorno Esperado")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(
        df_resultados["Ativo"],
        df_resultados["Retorno Esperado (%)"]
    )

    ax2.set_title("Retorno Esperado pelo CAPM")

    ax2.set_ylabel("Retorno (%)")

    st.pyplot(fig2)
