import pandas as pd
from pathlib import Path

DATA_PATH = Path('data/Movies_US.csv')

def load_and_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Put your CSV there.")
    df = pd.read_csv(path, encoding='latin-1')
    # Convert obvious numeric columns; non-numerics become NaN and will be dropped later if needed
    numeric_columns = [
        'Adjusted Gross ($mill)','Budget ($mill)','Gross ($mill)',
        'IMDb Rating','MovieLens Rating','Runtime (min)','US ($mill)'
    ]
    for c in numeric_columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

if __name__ == '__main__':
    df = load_and_clean(DATA_PATH)
    print('Rows, Cols:', df.shape)
    print('Preview:\n', df.head())