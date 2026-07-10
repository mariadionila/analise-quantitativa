import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from ui import apply_theme, hero


st.set_page_config(page_title="Validação Temporal", layout="wide")
apply_theme()

hero(
    "Validação Temporal",
    "Use divisão sequencial da série para avaliar se o modelo mantém desempenho fora da amostra.",
    "Séries temporais",
)

ativo = st.selectbox(
    "Selecione o ativo",
    ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"],
)

if st.button("Executar Validação"):
    with st.spinner("Baixando dados..."):
        dados = yf.download(ativo, start="2020-01-01", auto_adjust=True, progress=False)

    if dados.empty:
        st.error("Não foi possível obter os dados do ativo.")
        st.stop()

    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.get_level_values(0)

    dados["Retorno"] = dados["Close"].pct_change()
    dados["Lag_1"] = dados["Retorno"].shift(1)
    dados["Lag_2"] = dados["Retorno"].shift(2)
    dados["Lag_3"] = dados["Retorno"].shift(3)
    dados = dados.dropna()

    X = dados[["Lag_1", "Lag_2", "Lag_3"]]
    y = dados["Retorno"]
    tscv = TimeSeriesSplit(n_splits=5)

    resultados = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        modelo = RandomForestRegressor(n_estimators=200, random_state=42)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, pred)

        resultados.append({"Fold": fold, "MAE": mae})

    resultados = pd.DataFrame(resultados)

    st.subheader("Resultado por Fold")
    st.dataframe(resultados, use_container_width=True)
    st.metric("MAE Médio", f"{resultados['MAE'].mean():.6f}")
    st.line_chart(resultados.set_index("Fold"))
