# LoanVet

LoanVet is a machine learning system for credit risk classification built with Python and scikit-learn.

## Project Structure

- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for exploratory data analysis and modelling
- `src/` - Source code for data processing and model pipeline
- `models/` - Saved machine learning models
- `reports/` - Visualisations and reports

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

2. Create and activate a Python virtual environment

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Import and clean the raw dataset:
    ```bash
    python src/import_csv_to_sqlite.py
    python src/data_cleaning.py
    ```

## Usage

Once the cleaned dataset has been saved to data/loanvet.db, you can run the notebooks for EDA, model training, and evaluation:
    ```bash
    jupyter notebook
    ```

Navigate to the notebooks/ folder and run the analysis step by step.

