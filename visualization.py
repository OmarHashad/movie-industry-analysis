import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path('data/Movies_US.csv')

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put your CSV there.")
    df = pd.read_csv(DATA_PATH, encoding='latin-1')

    numeric_columns = [
        'Adjusted Gross ($mill)','Budget ($mill)','Gross ($mill)',
        'IMDb Rating','MovieLens Rating','Runtime (min)','US ($mill)'
    ]
    for col in numeric_columns:
        if col in df.columns:
            # histograms
            plt.figure()
            df[col].dropna().hist(bins=30)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col); plt.ylabel('Frequency')
            plt.savefig(f'results/plots/hist_{col.replace(" ","_").replace("($mill)","mill")}.png')
            plt.close()

    # scatter IMDb vs MovieLens if both exist
    if 'IMDb Rating' in df.columns and 'MovieLens Rating' in df.columns:
        plt.figure()
        plt.scatter(df['IMDb Rating'], df['MovieLens Rating'])
        plt.title('IMDb Rating vs MovieLens Rating')
        plt.xlabel('IMDb Rating'); plt.ylabel('MovieLens Rating')
        plt.savefig('results/plots/scatter_imdb_movielens.png')
        plt.close()

if __name__ == '__main__':
    main()