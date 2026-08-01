import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import riskfolio as rp

st.set_page_config(
    page_title="Otimização de Carteira",
    layout="wide"
)

st.title("Otimização de Carteiras")
st.markdown("## Alocação Ótima de Ativos via Teoria Moderna de Carteiras")

st.markdown(
    "Construa carteiras eficientes a partir dos ativos selecionados, "
    "utilizando **Markowitz (Média-Variância)**, **HRP (Hierarchical Risk Parity)** "
    "ou **Black-Litterman**."
)

st.sidebar.header("Parâmetros da Otimização")

lista_ativos = [
    "PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA",
    "WEGE3.SA","BBAS3.SA","MGLU3.SA","SUZB3.SA","JBSS3.SA",
    "RENT3.SA","LREN3.SA"
]

ativos = st.sidebar.multiselect(
    "Selecione os ativos",
    lista_ativos,
    default=["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA","WEGE3.SA","BBAS3.SA"]
)

data_inicio = st.sidebar.date_input("Data Inicial", pd.to_datetime("2023-01-01"))
data_fim = st.sidebar.date_input("Data Final", pd.to_datetime("today"))

taxa_livre = st.sidebar.number_input("Taxa Livre de Risco (% a.a.)", value=10.75) / 100

st.sidebar.markdown("## Modelo de Otimização")

modelo_escolhido = st.sidebar.selectbox(
    "Método",
    [
        "Markowitz (Média-Variância)",
        "HRP (Hierarchical Risk Parity)",
        "Black-Litterman"
    ]
)

if modelo_escolhido == "Markowitz (Média-Variância)":

    objetivo = st.sidebar.selectbox(
        "Objetivo",
        ["Sharpe Máximo", "Risco Mínimo"]
    )

if modelo_escolhido == "Black-Litterman":

    st.sidebar.markdown("### Visão do Investidor")

    ativo_visao = st.sidebar.selectbox(
        "Ativo com visão diferenciada",
        ativos if ativos else lista_ativos
    )

    retorno_visao = st.sidebar.number_input(
        "Retorno esperado anual para o ativo (%)",
        value=20.0
    ) / 100

st.sidebar.markdown("### Ativos analisados")
for a in ativos:
    st.sidebar.write(a)

if st.sidebar.button("Executar Otimização"):

    if len(ativos) < 2:
        st.warning("Selecione pelo menos 2 ativos para otimizar a carteira.")
        st.stop()

    precos = pd.DataFrame()

    for ativo in ativos:
        try:
            dados = yf.download(ativo, start=data_inicio, end=data_fim, progress=False, auto_adjust=True)

            close = dados["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            precos[ativo] = close

        except Exception as e:
            st.error(f"Erro ao baixar {ativo}: {e}")

    precos = precos.dropna()

    if precos.empty or len(precos) < 30:
        st.warning("Dados insuficientes para o período e ativos selecionados.")
        st.stop()

    retornos = precos.pct_change().dropna()

    # otimização

    try:

        if modelo_escolhido == "Markowitz (Média-Variância)":

            port = rp.Portfolio(returns=retornos)
            port.assets_stats(method_mu="hist", method_cov="hist")

            obj = "Sharpe" if objetivo == "Sharpe Máximo" else "MinRisk"

            pesos = port.optimization(
                model="Classic",
                rm="MV",
                obj=obj,
                rf=taxa_livre / 252,
                l=0,
                hist=True
            )

            fronteira = port.efficient_frontier(
                model="Classic",
                rm="MV",
                points=30,
                rf=taxa_livre / 252,
                hist=True
            )

        elif modelo_escolhido == "HRP (Hierarchical Risk Parity)":

            port = rp.HCPortfolio(returns=retornos)

            pesos = port.optimization(
                model="HRP",
                codependence="pearson",
                rm="MV",
                rf=taxa_livre / 252,
                linkage="single"
            )

            fronteira = None

        else:

            port = rp.Portfolio(returns=retornos)
            port.assets_stats(method_mu="hist", method_cov="hist")

            P = pd.DataFrame(np.zeros((1, len(ativos))), columns=retornos.columns)
            P.loc[0, ativo_visao] = 1

            Q = pd.DataFrame([retorno_visao / 252])

            port.blacklitterman_stats(
                P=P,
                Q=Q,
                rf=taxa_livre / 252,
                w=None,
                delta=None,
                eq=True
            )

            pesos = port.optimization(
                model="BL",
                rm="MV",
                obj="Sharpe",
                rf=taxa_livre / 252,
                l=0,
                hist=False
            )

            fronteira = None

    except Exception as e:
        st.error(f"Erro na otimização: {e}")
        st.stop()

    if pesos is None or pesos.empty:
        st.warning("Não foi possível calcular os pesos da carteira.")
        st.stop()

    # métricas da carteira

    pesos_vetor = pesos["weights"].values

    retorno_medio_diario = retornos.mean().values
    retorno_esperado = np.dot(pesos_vetor, retorno_medio_diario) * 252

    cov_matriz = retornos.cov() * 252
    volatilidade = np.sqrt(pesos_vetor.T @ cov_matriz.values @ pesos_vetor)

    sharpe = (retorno_esperado - taxa_livre) / volatilidade if volatilidade != 0 else 0

    pesos_iguais = np.repeat(1 / len(ativos), len(ativos))
    retorno_igual = np.dot(pesos_iguais, retorno_medio_diario) * 252
    volatilidade_igual = np.sqrt(pesos_iguais.T @ cov_matriz.values @ pesos_iguais)
    sharpe_igual = (retorno_igual - taxa_livre) / volatilidade_igual if volatilidade_igual != 0 else 0

    st.success("Otimização concluída!")

    st.divider()

    st.subheader("Indicadores da Carteira Ótima")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ativos", len(ativos))
    c2.metric("Retorno Esperado (a.a.)", f"{retorno_esperado*100:.2f}%")
    c3.metric("Volatilidade (a.a.)", f"{volatilidade*100:.2f}%")
    c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Pesos Ótimos")

        df_pesos = pesos.copy()
        df_pesos.columns = ["Peso"]
        df_pesos["Peso"] = df_pesos["Peso"] * 100
        df_pesos = df_pesos.sort_values("Peso", ascending=False)

        st.dataframe(
            df_pesos.style.background_gradient(
                cmap="Blues",
                subset=["Peso"]
            ).format({"Peso": "{:.2f}%"}),
            use_container_width=True
        )

    with col2:

        st.subheader("Alocação")

        fig_pizza = rp.plot_pie(
            w=pesos,
            title="Alocação da Carteira",
            height=6,
            width=10,
            ax=None
        )

        st.pyplot(fig_pizza.figure)

    st.divider()

    if fronteira is not None:

        st.subheader("Fronteira Eficiente")

        plt.close("all")

        fig_front, ax_front = plt.subplots(figsize=(10, 6))

        rp.plot_frontier(
            w_frontier=fronteira,
            mu=port.mu,
            cov=port.cov,
            returns=retornos,
            rm="MV",
            rf=taxa_livre / 252,
            alpha=0.05,
            cmap="viridis",
            w=pesos,
            label="Carteira Selecionada",
            marker="*",
            s=18,
            c="red",
            ax=ax_front
        )

        fig_front.subplots_adjust(
            left=0.10,
            right=0.82,
            bottom=0.12,
            top=0.90
        )

        st.pyplot(fig_front)

        plt.close(fig_front)

        st.divider()

    st.subheader("Comparação: Carteira Ótima x Igual Peso")

    df_comparacao = pd.DataFrame({
        "Carteira": ["Ótima", "Igual Peso"],
        "Retorno Esperado (%)": [retorno_esperado*100, retorno_igual*100],
        "Volatilidade (%)": [volatilidade*100, volatilidade_igual*100],
        "Sharpe": [sharpe, sharpe_igual]
    })

    st.dataframe(df_comparacao, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    largura = 0.35
    posicoes = np.arange(2)

    ax2.bar(posicoes - largura/2, df_comparacao["Retorno Esperado (%)"], largura, label="Retorno (%)")
    ax2.bar(posicoes + largura/2, df_comparacao["Volatilidade (%)"], largura, label="Volatilidade (%)")

    ax2.set_xticks(posicoes)
    ax2.set_xticklabels(df_comparacao["Carteira"])
    ax2.legend()
    ax2.set_title("Retorno x Risco")

    st.pyplot(fig2)
