import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from ui import apply_theme, hero, style_axis


st.set_page_config(page_title="IA e Predição", layout="wide")
apply_theme()

hero(
    "Inteligência Artificial e Predição",
    "Compare modelos tradicionais de aprendizado de máquina para prever retornos futuros a partir de defasagens da série.",
    "Machine Learning",
)

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
    "LREN3.SA",
]

ativo = st.sidebar.selectbox("Selecione o ativo", ativos)
modelo_escolhido = st.sidebar.selectbox("Modelo", ["Random Forest", "XGBoost", "LightGBM"])
data_inicio = st.sidebar.date_input("Data inicial", pd.to_datetime("2020-01-01"))
data_fim = st.sidebar.date_input("Data final", pd.to_datetime("today"))

if st.sidebar.button("Executar Previsão"):
    dados = yf.download(
        ativo,
        start=data_inicio,
        end=data_fim,
        progress=False,
        auto_adjust=True,
    )

    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.get_level_values(0)

    dados = dados[["Close"]].copy()
    dados["Retorno"] = dados["Close"].pct_change()

    for i in range(1, 6):
        dados[f"Lag_{i}"] = dados["Retorno"].shift(i)

    dados["Media_5"] = dados["Retorno"].rolling(5).mean()
    dados["Media_10"] = dados["Retorno"].rolling(10).mean()
    dados.dropna(inplace=True)

    features = ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5", "Media_5", "Media_10"]
    X = dados[features].copy()
    y = dados["Retorno"]
    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in X.columns]

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    if modelo_escolhido == "Random Forest":
        modelo = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
    elif modelo_escolhido == "XGBoost":
        modelo = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
    else:
        modelo = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=-1,
        )
        modelo.fit(X_train.to_numpy(), y_train.to_numpy())
        previsoes = modelo.predict(X_test.to_numpy())

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
    ax.plot(y_test.values, label="Real", color="#22d3ee")
    ax.plot(previsoes, label="Previsto", color="#a3e635")
    ax.legend()
    style_axis(fig, ax)
    st.pyplot(fig)

    ultima_linha = X.tail(1)

    if modelo_escolhido == "LightGBM":
        proximo_retorno = modelo.predict(ultima_linha.to_numpy())[0]
    else:
        proximo_retorno = modelo.predict(ultima_linha)[0]

    st.subheader("Próximo Retorno Previsto")
    st.success(f"{proximo_retorno * 100:.2f}%")

    if modelo_escolhido != "LightGBM" and hasattr(modelo, "feature_importances_"):
        importancia = (
            pd.DataFrame(
                {
                    "Variável": X.columns,
                    "Importância": modelo.feature_importances_,
                }
            )
            .sort_values("Importância", ascending=False)
        )

        st.subheader("Importância das Variáveis")
        st.dataframe(importancia, use_container_width=True)
