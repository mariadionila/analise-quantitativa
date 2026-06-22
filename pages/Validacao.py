import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import (
    TimeSeriesSplit
)

from .deep_learning import (
    LSTMModel,
    GRUModel,
    train_model,
    predict
)


def time_series_validation(
    X,
    y,
    model_type="LSTM",
    splits=5
):

    tscv = TimeSeriesSplit(
        n_splits=splits
    )

    metrics = []

    for train_idx, test_idx in tscv.split(X):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        if model_type == "LSTM":

            model = LSTMModel()

        else:

            model = GRUModel()

        model = train_model(
            model,
            X_train,
            y_train
        )

        preds = predict(
            model,
            X_test
        )

        mse = mean_squared_error(
            y_test,
            preds
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_test,
            preds
        )

        r2 = r2_score(
            y_test,
            preds
        )

        metrics.append({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

    return {
        "mse": np.mean(
            [m["mse"] for m in metrics]
        ),
        "rmse": np.mean(
            [m["rmse"] for m in metrics]
        ),
        "mae": np.mean(
            [m["mae"] for m in metrics]
        ),
        "r2": np.mean(
            [m["r2"] for m in metrics]
        )
    }