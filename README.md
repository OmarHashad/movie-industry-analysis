# 🎬 Movie Industry Data Analysis

Analyze U.S. movie data and build ML models to predict **Gross ($M)**.

## Project Structure
```
movie-industry-analysis/
├── data/                 # place Movies_US.csv here (not included)
├── results/              # metrics & saved plots
├── src/                  # scripts
├── README.md
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
# 1) Put your dataset at: data/Movies_US.csv
python src/preprocess.py
python src/visualization.py
python src/models.py
```

## Notes
- Models compared: Linear Regression, SVM (linear/poly/rbf), MLPRegressor.
- Two scalers tried: StandardScaler, MinMaxScaler.
- Outputs: printed metrics and (optionally) saved plots/CSV to `results/`.
