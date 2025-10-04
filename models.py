import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = Path('data/Movies_US.csv')
TARGET = 'Gross ($mill)'

def load_numeric(df: pd.DataFrame):
    # keep only numeric columns
    num = df.select_dtypes(include=['number']).copy()
    if TARGET not in num.columns:
        raise ValueError(f'Target column {TARGET!r} not found or not numeric.')
    X = num.drop(columns=[TARGET]).dropna(axis=1, how='all')
    y = num[TARGET]
    # Drop rows with any NaNs in features/target
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    return X, y

def evaluate(X, y, scaler_name, scaler):
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'SVM (linear)': SVR(kernel='linear'),
        'SVM (poly)': SVR(kernel='poly', degree=3),
        'SVM (rbf)': SVR(kernel='rbf'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f'{scaler_name} | {name}: R2={r2:.3f}, MSE={mse:.6f}')
        rows.append({'scaler': scaler_name, 'model': name, 'r2': r2, 'mse': mse})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put your CSV there.")
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    X, y = load_numeric(df)

    out = []
    out.append(evaluate(X, y, 'StandardScaler', StandardScaler()))
    out.append(evaluate(X, y, 'MinMaxScaler', MinMaxScaler()))

    results = pd.concat(out, ignore_index=True)
    results.to_csv('results/metrics_summary.csv', index=False)
    print('\nSaved results to results/metrics_summary.csv')