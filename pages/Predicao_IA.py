import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="IA e Predição",
    layout="wide"
)

st.title("Inteligência Artificial e Predição")

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

data_inicio = st.sidebar.date_input(
    "Data Inicial",
    pd.to_datetime("2020-01-01")
)

data_fim = st.sidebar.date_input(
    "Data Final",
    pd.to_datetime("today")
)

if st.sidebar.button("Executar Previsão"):

    dados = yf.download(
        ativo,
        start=data_inicio,
        end=data_fim,
        progress=False
    )

    dados = dados[["Close"]]

    dados["Retorno"] = dados["Close"].pct_change()

    dados["Lag_1"] = dados["Retorno"].shift(1)
    dados["Lag_2"] = dados["Retorno"].shift(2)
    dados["Lag_3"] = dados["Retorno"].shift(3)
    dados["Lag_4"] = dados["Retorno"].shift(4)
    dados["Lag_5"] = dados["Retorno"].shift(5)

    dados["Media_5"] = dados["Retorno"].rolling(5).mean()
    dados["Media_10"] = dados["Retorno"].rolling(10).mean()

    dados = dados.dropna()

    X = dados[
        [
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_4",
            "Lag_5",
            "Media_5",
            "Media_10"
        ]
    ]

    y = dados["Retorno"]

    split = int(len(dados) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

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
            max_depth=5,
            random_state=42
        )

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, previsoes)
    rmse = np.sqrt(mean_squared_error(y_test, previsoes))
    r2 = r2_score(y_test, previsoes)

    st.subheader("Métricas")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.6f}")
    c2.metric("RMSE", f"{rmse:.6f}")
    c3.metric("R²", f"{r2:.4f}")

    st.subheader("Previsão x Real")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        y_test.values,
        label="Real"
    )

    ax.plot(
        previsoes,
        label="Previsto"
    )

    ax.legend()

    st.pyplot(fig)

    ultima_linha = X.tail(1)

    proximo_retorno = modelo.predict(
        ultima_linha
    )[0]

    st.subheader("Próximo Retorno Previsto")

    st.success(
        f"{proximo_retorno * 100:.2f}%"
    )

    if hasattr(modelo, "feature_importances_"):

        importancia = pd.DataFrame(
            {
                "Variável": X.columns,
                "Importância": modelo.feature_importances_
            }
        ).sort_values(
            "Importância",
            ascending=False
        )

        st.subheader("Importância das Variáveis")

        st.dataframe(importancia)