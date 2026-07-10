import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="Deep Learning",
    layout="wide"
)

st.title("Deep Learning para Séries Temporais")

ativo = st.selectbox(
    "Selecione o Ativo",
    [
        "PETR4.SA",
        "VALE3.SA",
        "ITUB4.SA",
        "WEGE3.SA",
        "BBAS3.SA"
    ]
)

modelo_tipo = st.selectbox(
    "Modelo",
    [
        "LSTM",
        "GRU"
    ]
)

janela = st.slider(
    "Janela Temporal",
    10,
    60,
    20
)

epocas = st.slider(
    "Épocas",
    10,
    200,
    50
)

if st.button("Treinar Modelo"):

    dados = yf.download(
        ativo,
        start="2018-01-01",
        progress=False
    )

    serie = dados["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    serie = scaler.fit_transform(serie)

    X = []
    y = []

    for i in range(janela, len(serie)):
        X.append(serie[i-janela:i])
        y.append(serie[i])

    X = np.array(X)
    y = np.array(y)

    corte = int(len(X) * 0.8)

    X_train = X[:corte]
    X_test = X[corte:]

    y_train = y[:corte]
    y_test = y[corte:]

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    class ModeloLSTM(nn.Module):

        def __init__(self):

            super().__init__()

            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=64,
                batch_first=True
            )

            self.fc = nn.Linear(
                64,
                1
            )

        def forward(self, x):

            out, _ = self.lstm(x)

            out = out[:, -1, :]

            return self.fc(out)

    class ModeloGRU(nn.Module):

        def __init__(self):

            super().__init__()

            self.gru = nn.GRU(
                input_size=1,
                hidden_size=64,
                batch_first=True
            )

            self.fc = nn.Linear(
                64,
                1
            )

        def forward(self, x):

            out, _ = self.gru(x)

            out = out[:, -1, :]

            return self.fc(out)

    if modelo_tipo == "LSTM":
        modelo = ModeloLSTM()
    else:
        modelo = ModeloGRU()

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        modelo.parameters(),
        lr=0.001
    )

    barra = st.progress(0)

    for epoch in range(epocas):

        modelo.train()

        previsao = modelo(X_train)

        loss = loss_fn(
            previsao,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        barra.progress(
            (epoch + 1) / epocas
        )

    modelo.eval()

    with torch.no_grad():

        predicoes = modelo(X_test)

    predicoes = predicoes.numpy()

    predicoes = scaler.inverse_transform(
        predicoes
    )

    y_real = scaler.inverse_transform(
        y_test.numpy()
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            predicoes
        )
    )

    mae = mean_absolute_error(
        y_real,
        predicoes
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    col2.metric(
        "MAE",
        f"{mae:.2f}"
    )

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        y_real,
        label="Real"
    )

    ax.plot(
        predicoes,
        label="Previsto"
    )

    ax.set_title(
        f"{modelo_tipo} - {ativo}"
    )

    ax.legend()

    st.pyplot(fig)

    ultimo = torch.tensor(
        X[-1:].copy(),
        dtype=torch.float32
    )

    proximo = modelo(
        ultimo
    ).detach().numpy()

    proximo = scaler.inverse_transform(
        proximo
    )

    st.success(
        f"Próximo preço previsto: R$ {proximo[0][0]:.2f}"
    )

    resultados = pd.DataFrame(
        {
            "Preço Real": y_real.flatten(),
            "Preço Previsto": predicoes.flatten()
        }
    )

    st.dataframe(
        resultados.tail(20)
    )