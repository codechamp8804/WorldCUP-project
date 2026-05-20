# World Cup Match Outcome Predictor

Predict whether the home team will win, lose, or draw using historical international football results.

The project uses:

- Python
- Pandas
- Scikit-learn
- Streamlit
- Kaggle dataset: `martj42/international-football-results-from-1872-to-2017`

## 1. Set up

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

Option A: download from Kaggle manually:

1. Open https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
2. Download the dataset.
3. Put `results.csv` inside the `data/` folder.

Option B: use the Kaggle API through `kagglehub`:

```bash
python train_model.py --download
```

This requires Kaggle credentials to be configured on your machine.

## 3. Train the model

```bash
python train_model.py --data data/results.csv
```

The trained model is saved to `artifacts/model.joblib`.
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py --data data/results.csv
streamlit run app.py

## 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Notes

Football is noisy, so the goal is not perfect accuracy. The value of this project is the clean machine-learning workflow and the thoughtful feature engineering:

- recent form from the last 5 matches
- goals scored and conceded recently
- historical win rates
- neutral-site and home/away context
- tournament and location context

The model predicts the result from the home team's perspective:

- `Home Win`
- `Draw`
- `Away Win`
