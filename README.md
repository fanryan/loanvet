# LoanVet

LoanVet is a comprehensive end-to-end machine learning system developed to predict credit risk based on financial and behavioural data. It utilises advanced feature engineering alongside powerful classification algorithms—such as Random Forest, XGBoost, and LightGBM—to deliver accurate and robust predictions. The system features a FastAPI backend that serves predictions through a RESTful API, enabling seamless integration with dashboards, web applications, or other frontend interfaces.

Try the live app here: https://fan-loanvet.streamlit.app

## Project Structure

- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for exploratory data analysis and modelling
- `src/` - Source code for data import, cleaning, and feature engineering
- `models/` - Saved pipelines, evaluation metrics, and feature importance artefacts
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
   pip install -r dev-requirements.txt
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

## Model Management

- Trained models are saved as `.joblib` files in the `models/` directory.
- Due to size and reproducibility considerations, the final model files are **not included** in this repository.
- To reproduce the final trained models, please run the notebooks in order of number.
- Following this sequence will generate and save the final models automatically in the `models/` folder.

## Reports

Final evaluation reports are saved as HTML files under the `reports/` directory for easy sharing and review.  
For example:  
`reports/final_evaluation_report.html`

Open these files in a web browser to view comprehensive model performance summaries, ROC and Precision-Recall curves, threshold tuning, and error analysis visualisations.

## Deployment

### Run Backend API Locally

To start the FastAPI backend server locally, run:

```bash
uvicorn src.api.app:app --reload
```
This will launch the API at:  
[http://localhost:8000](http://localhost:8000)

### Run the Streamlit Frontend

To launch the Streamlit app, run:

```bash
streamlit run src/streamlit_app.py
```
  
## Evaluation Summary

All models were trained on the fully engineered dataset using 5-fold stratified cross-validation.  
The metrics below reflect classification performance under significant class imbalance.

### Baseline Model — Logistic Regression

- **ROC-AUC:** 0.8556 
- **PR-AUC:** 0.3706
- **F1 Score:** 0.3345

The baseline logistic regression model demonstrates reasonable ability to rank positive and negative cases but struggles with recall and precision on the minority positive class.

### Random Forest (Threshold = 0.3644)

- **ROC-AUC:** 0.8434  
- **PR-AUC:** 0.3437  
- **F1 Score:** 0.4204  

Random Forest shows improvement in F1 score compared to baseline, indicating a better balance between precision and recall, but its ranking ability (ROC-AUC) is slightly lower than other ensemble methods.

### XGBoost (Threshold = 0.2268)

- **ROC-AUC:** 0.8643  
- **PR-AUC:** 0.4038  
- **F1 Score:** 0.4589  

XGBoost delivers the best overall performance across all key metrics, showing stronger discrimination and better handling of the imbalanced positive class. This model achieves the highest precision-recall balance, making it the optimal choice for credit risk prediction.

### LightGBM (Threshold = 0.8077)

- **ROC-AUC:** 0.8633  
- **PR-AUC:** 0.3986  
- **F1 Score:** 0.4552  

LightGBM performs comparably to XGBoost with near-equivalent ROC-AUC and PR-AUC scores, maintaining strong predictive ability and balance.

### Summary

The advanced ensemble models demonstrate clear improvements over the logistic regression baseline in both ranking and minority class prediction. Among the advanced models, XGBoost provides the best trade-off between precision and recall, which is critical in credit risk classification. Therefore, **XGBoost was selected as the final model for deployment in the Streamlit application**, balancing performance and interpretability for real-world use.

## Future Work and Improvements

While the current XGBoost model demonstrates strong performance, several avenues exist to further enhance LoanVet’s accuracy, robustness, and usability:

### 1. **Hyperparameter Optimisation**
- Implement more extensive hyperparameter tuning using techniques such as Bayesian Optimisation or Genetic Algorithms to potentially improve model performance beyond default or grid search settings.
- Experiment with ensemble stacking or blending of models (e.g., combining XGBoost and LightGBM) to leverage complementary strengths.

### 2. **Feature Engineering Enhancements**
- Explore additional feature transformations, interaction terms, and domain-driven feature creation, especially leveraging behavioural data and temporal trends.
- Incorporate external data sources (e.g., credit bureau scores, macroeconomic indicators) to provide richer context for risk prediction.

### 3. **Handling Class Imbalance**
- Investigate alternative imbalance handling techniques such as SMOTE, ADASYN, or focal loss within tree-based models to better capture rare default cases.
- Adjust threshold tuning dynamically based on business objectives or cost-sensitive learning frameworks.

### 4. **Model Explainability and Interpretability**
- Integrate explainability tools like LIME to provide transparent model insights for stakeholders, enhancing trust and regulatory compliance.
- Develop user-friendly visual dashboards highlighting feature importance and risk drivers for individual predictions.

### 5. **Deployment and Monitoring**
- Build automated pipelines for model retraining and monitoring to adapt to data drift and maintain performance over time.
- Set up alerting for performance degradation or data anomalies during production use.

### 6. **Broader Evaluation Metrics**
- Expand evaluation to include business-centric metrics such as expected loss, cost savings, and portfolio impact.
- Conduct fairness and bias assessments to ensure equitable predictions across demographic groups.

### 7. **Integration and Scalability**
- Optimise backend API performance and scale the system for higher throughput or batch predictions.
- Extend frontend capabilities with richer interactivity and personalised user experiences.

Pursuing these improvements will make LoanVet a more robust, interpretable, and business-aligned credit risk tool, capable of adapting to evolving data and operational needs.

## Contributing
Thank you for exploring LoanVet!
If you encounter any issues or have ideas for improvements, please feel free to open an issue or submit a pull request.
Your contributions and feedback are highly appreciated.

— Fan Ryan
