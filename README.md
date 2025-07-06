# LoanVet

LoanVet is a machine learning system for credit risk classification built with Python and scikit-learn.

## Project Structure

- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for exploratory data analysis and modelling
- `src/` - Source code for data import, cleaning, and feature engineering
- `models/` - Saved pipelines, evaluation metrics, and feature importance artifacts
- `reports/` - Visualisations and reports
- `features.md` - Documentation of engineered features and transformations

## Dataset

This project uses the [Give Me Some Credit dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) from Kaggle.

### 📥 Download Instructions

1. Visit the dataset page:  
   https://www.kaggle.com/competitions/GiveMeSomeCredit/data

2. Download the file named:  
   `cs-training.csv`

3. Rename it to:  
   `credit_train.csv`

4. Place it in the following folder:  
   `data/raw/credit_train.csv`

## Setup

1. Clone this repository and navigate into the project folder:
   ```bash
   git clone https://github.com/fanryan/LoanVet.git
   cd LoanVet
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Import and clean the raw dataset, then perform feature engineering:
   ```bash
   python src/import_csv_to_sqlite.py
   python src/data_cleaning.py
   python src/feature_engineering.py
   ```

## Features Documentation

All engineered features and transformations are documented in the `features.md` file.  
This file explains how raw variables are transformed, aggregated, binned, or combined into new features used for modelling.

## Usage

Once the cleaned and engineered dataset has been saved to `data/loanvet.db`, you can run the notebooks for exploratory data analysis, model training, and evaluation:

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and open the notebooks.  
**Tip:** In VS Code, press `Cmd + Shift + V` (Mac) or `Ctrl + Shift + V` (Windows/Linux) to preview Markdown files like `features.md` directly.

## Saved Models

Trained models and outputs are saved in the `models/` directory.

The artifacts support downstream reuse in dashboards (e.g. Streamlit) and inference scripts.

## Additional Notes

- The dataset is imbalanced; some models use class weighting to address this.
- Feature engineering is central to improving model performance — review `features.md` regularly.
- Keep dependencies up to date by updating `requirements.txt`.
- You can extend or modify feature engineering by editing `src/feature_engineering.py`.
- Models are saved in the `models/` directory after training for easy reuse.

---

Feel free to open issues or submit pull requests for improvements!  
— Ryan Fan
EOF
