import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Validação",
    layout="wide"
)

st.title("Validação Temporal")

ativo = st.selectbox(
    "Ativo",
    [
        "PETR4.SA",
        "VALE3.SA",
        "ITUB4.SA",
        "WEGE3.SA"
    ]
)

if st.button("Executar Validação"):

    dados = yf.download(
        ativo,
        start="2020-01-01",
        progress=False
    )

    dados["Retorno"] = (
        dados["Close"].pct_change()
    )

    dados["Lag_1"] = dados["Retorno"].shift(1)
    dados["Lag_2"] = dados["Retorno"].shift(2)
    dados["Lag_3"] = dados["Retorno"].shift(3)

    dados = dados.dropna()

    X = dados[
        [
            "Lag_1",
            "Lag_2",
            "Lag_3"
        ]
    ]

    y = dados["Retorno"]

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    resultados = []

    for fold, (train, test) in enumerate(tscv.split(X)):

        X_train = X.iloc[train]
        X_test = X.iloc[test]

        y_train = y.iloc[train]
        y_test = y.iloc[test]

        modelo = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        modelo.fit(
            X_train,
            y_train
        )

        pred = modelo.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            pred
        )

        resultados.append(
            {
                "Fold": fold + 1,
                "MAE": mae
            }
        )

    resultados = pd.DataFrame(
        resultados
    )

    st.dataframe(resultados)

    st.metric(
        "MAE Médio",
        f"{resultados['MAE'].mean():.6f}"
    )