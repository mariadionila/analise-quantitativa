import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predição com IA",
    layout="wide"
)

st.title("Inteligência Artificial para Predição de Retornos")

ativos = [
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

ativo = st.sidebar.selectbox(
    "Selecione o ativo",
    ativos
)

modelo_escolhido = st.sidebar.selectbox(
    "Modelo",
    [
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ]
)

inicio = st.sidebar.date_input(
    "Data Inicial",
    pd.to_datetime("2020-01-01")
)

fim = st.sidebar.date_input(
    "Data Final",
    pd.to_datetime("today")
)

if st.sidebar.button("Executar Predição"):

    dados = yf.download(
        ativo,
        start=inicio,
        end=fim,
        progress=False
    )

    if dados.empty:
        st.error("Sem dados disponíveis")
        st.stop()

    df = dados.copy()

    df["Retorno"] = df["Close"].pct_change()

    # FEATURES

    df["ret_1"] = df["Retorno"].shift(1)

    df["ret_5"] = (
        df["Retorno"]
        .rolling(5)
        .mean()
    )

    df["ret_20"] = (
        df["Retorno"]
        .rolling(20)
        .mean()
    )

    df["vol_5"] = (
        df["Retorno"]
        .rolling(5)
        .std()
    )

    df["vol_20"] = (
        df["Retorno"]
        .rolling(20)
        .std()
    )

    df["ma_5"] = (
        df["Close"]
        .rolling(5)
        .mean()
    )

    df["ma_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["ma_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # TARGET

    df["Target"] = (
        df["Retorno"]
        .shift(-1)
    )

    df = df.dropna()

    features = [
        "ret_1",
        "ret_5",
        "ret_20",
        "vol_5",
        "vol_20",
        "ma_5",
        "ma_20",
        "ma_50"
    ]

    X = df[features]

    y = df["Target"]

    split = int(len(df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # MODELOS

    if modelo_escolhido == "Random Forest":

        modelo = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )

    elif modelo_escolhido == "XGBoost":

        modelo = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

    else:

        modelo = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        previsoes
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            previsoes
        )
    )

    r2 = r2_score(
        y_test,
        previsoes
    )

    st.subheader("Métricas")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{mae:.6f}"
    )

    c2.metric(
        "RMSE",
        f"{rmse:.6f}"
    )

    c3.metric(
        "R²",
        f"{r2:.4f}"
    )

    st.subheader("Real x Previsto")

    resultado = pd.DataFrame({
        "Real": y_test,
        "Previsto": previsoes
    })

    st.dataframe(
        resultado.tail(20),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        resultado.index,
        resultado["Real"],
        label="Real"
    )

    ax.plot(
        resultado.index,
        resultado["Previsto"],
        label="Previsto"
    )

    ax.legend()

    st.pyplot(fig)

    st.subheader(
        "Importância das Variáveis"
    )

    importancia = pd.DataFrame({
        "Variável": features,
        "Importância":
            modelo.feature_importances_
    })

    importancia = importancia.sort_values(
        "Importância",
        ascending=False
    )

    st.dataframe(
        importancia,
        use_container_width=True
    )

    fig2, ax2 = plt.subplots(
        figsize=(10,5)
    )

    ax2.bar(
        importancia["Variável"],
        importancia["Importância"]
    )

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    ultimo = X.tail(1)

    proximo_retorno = modelo.predict(
        ultimo
    )[0]

    st.subheader(
        "Previsão Próximo Pregão"
    )

    st.success(
        f"Retorno previsto: {proximo_retorno:.4%}"
    )