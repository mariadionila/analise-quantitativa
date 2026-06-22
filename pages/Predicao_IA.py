import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_length=30):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.gru(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


def prepare_data(series, seq_length=30):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        np.array(series).reshape(-1, 1)
    )

    X, y = create_sequences(
        scaled,
        seq_length
    )

    return X, y, scaler


def train_model(
    model,
    X_train,
    y_train,
    epochs=50,
    lr=0.001
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        predictions = model(X_train)

        loss = criterion(
            predictions,
            y_train.reshape(-1, 1)
        )

        loss.backward()

        optimizer.step()

    return model


def predict(model, X):

    model.eval()

    with torch.no_grad():

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        preds = model(X)

    return preds.numpy()