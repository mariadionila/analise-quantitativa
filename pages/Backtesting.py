import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Backtesting",
    layout="wide"
)

st.title("Backtesting das Estratégias")

##############################################################

ticker = st.text_input("Ticker", "AAPL")

inicio = st.date_input(
    "Data Inicial",
    pd.to_datetime("2018-01-01")
)

fim = st.date_input(
    "Data Final",
    pd.to_datetime("today")
)

capital_inicial = st.number_input(
    "Capital Inicial",
    value=10000.0,
    step=1000.0
)

##############################################################

if st.button("Executar Backtesting"):

    dados = yf.download(
        ticker,
        start=inicio,
        end=fim
    )

    if dados.empty:
        st.error("Nenhum dado encontrado.")
        st.stop()

    close = dados["Close"]

    df = pd.DataFrame()

    ##########################################################

    for lag in range(1,11):
        df[f"lag_{lag}"] = close.shift(lag)

    df["target"] = close

    df.dropna(inplace=True)

    ##########################################################

    treino = int(len(df)*0.8)

    treino_df = df.iloc[:treino]
    teste_df = df.iloc[treino:]

    X_train = treino_df.drop("target", axis=1)
    y_train = treino_df["target"]

    X_test = teste_df.drop("target", axis=1)
    y_test = teste_df["target"]

    ##########################################################

    modelo = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    ##########################################################

    resultado = pd.DataFrame()

    resultado["Real"] = y_test.values
    resultado["Previsto"] = previsoes

    ##########################################################

    resultado["Sinal"] = np.where(
        resultado["Previsto"] > resultado["Real"],
        1,
        -1
    )

    ##########################################################

    retorno = resultado["Real"].pct_change().fillna(0)

    resultado["Retorno Mercado"] = retorno.values

    resultado["Retorno Estratégia"] = (
        resultado["Retorno Mercado"] *
        resultado["Sinal"].shift(1).fillna(0)
    )

    ##########################################################

    resultado["Capital"] = (
        1 + resultado["Retorno Estratégia"]
    ).cumprod() * capital_inicial

    resultado["BuyHold"] = (
        1 + resultado["Retorno Mercado"]
    ).cumprod() * capital_inicial

    ##########################################################

    retorno_total = (
        resultado["Capital"].iloc[-1] /
        capital_inicial - 1
    ) * 100

    retorno_buy_hold = (
        resultado["BuyHold"].iloc[-1] /
        capital_inicial - 1
    ) * 100

    ##########################################################

    volatilidade = resultado["Retorno Estratégia"].std()

    sharpe = 0

    if volatilidade != 0:

        sharpe = (
            resultado["Retorno Estratégia"].mean() /
            volatilidade
        ) * np.sqrt(252)

    ##########################################################

    pico = resultado["Capital"].cummax()

    drawdown = (
        resultado["Capital"] - pico
    ) / pico

    max_drawdown = drawdown.min() * 100

    ##########################################################

    direcao_real = np.where(
        resultado["Retorno Mercado"] > 0,
        1,
        0
    )

    direcao_prevista = np.where(
        resultado["Retorno Estratégia"] > 0,
        1,
        0
    )

    acuracia = accuracy_score(
        direcao_real,
        direcao_prevista
    )

    ##########################################################

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Capital Final",
        f"R$ {resultado['Capital'].iloc[-1]:,.2f}"
    )

    col2.metric(
        "Retorno (%)",
        f"{retorno_total:.2f}%"
    )

    col3.metric(
        "Sharpe",
        f"{sharpe:.2f}"
    )

    col4.metric(
        "Drawdown",
        f"{max_drawdown:.2f}%"
    )

    ##########################################################

    st.subheader("Comparação")

    fig, ax = plt.subplots(figsize=(14,6))

    ax.plot(
        resultado["Capital"],
        label="Estratégia"
    )

    ax.plot(
        resultado["BuyHold"],
        label="Buy & Hold"
    )

    ax.set_ylabel("Capital")

    ax.legend()

    st.pyplot(fig)

    ##########################################################

    st.subheader("Drawdown")

    fig2, ax2 = plt.subplots(figsize=(14,4))

    ax2.fill_between(
        range(len(drawdown)),
        drawdown,
        color="red",
        alpha=0.4
    )

    ax2.set_ylabel("Drawdown")

    st.pyplot(fig2)

    ##########################################################

    st.subheader("Tabela de Resultados")

    resumo = pd.DataFrame({

        "Indicador":[

            "Capital Inicial",
            "Capital Final",
            "Retorno Estratégia",
            "Retorno BuyHold",
            "Sharpe Ratio",
            "Drawdown Máximo",
            "Acurácia"

        ],

        "Valor":[

            capital_inicial,
            resultado["Capital"].iloc[-1],
            f"{retorno_total:.2f}%",
            f"{retorno_buy_hold:.2f}%",
            round(sharpe,2),
            f"{max_drawdown:.2f}%",
            f"{acuracia*100:.2f}%"

        ]

    })

    st.dataframe(
        resumo,
        use_container_width=True
    )

    ##########################################################

    st.subheader("Histórico das Operações")

    st.dataframe(
        resultado,
        use_container_width=True
    )