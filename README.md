# Tourism Package Prediction

MLOps project that predicts whether a customer will purchase the Wellness
Tourism Package. Runs entirely on GitHub — data lives in the repo,
training runs in GitHub Actions, the trained model is committed back
to the repo, and the Streamlit app runs in GitHub Codespaces.

## Setup

1. Push this repo to GitHub.
2. Drop `tourism.csv` into `tourism_project/data/`.
3. Push to `main`. The pipeline trains a model and commits it to
   `tourism_project/models/`.

No secrets required — the workflow uses the built-in `GITHUB_TOKEN`.

## Run the app in Codespaces

1. **Code → Codespaces → Create codespace on main**.
2. Wait ~1–2 min for the container to build and install deps.
3. In the Codespaces terminal:
   ```bash
   streamlit run tourism_project/deployment/app.py
   ```
4. When Codespaces prompts about port 8501, click **Open in Browser**.

## Run locally (optional)

```bash
pip install -r tourism_project/requirements.txt -r tourism_project/deployment/requirements.txt
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
streamlit run tourism_project/deployment/app.py
```

## CI/CD

`.github/workflows/pipeline.yml`: **data-prep → model-training →
commit-model**. Runs on every push to `main` except pushes that only
touch `tourism_project/models/**` (so the bot's own commit doesn't
retrigger the workflow).

MLflow uses a local sqlite backend at `mlruns/mlflow.db`. Browse a
downloaded run artifact with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
