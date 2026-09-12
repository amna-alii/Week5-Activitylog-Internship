# Week5-Activitylog-Internship
# Telco Customer Churn — Model Optimization & Final Selection (Week 5, Phase II)

Hyperparameter optimization and final model selection for the Telco Customer Churn prediction problem, performed as Phase II of a Week 5 internship task. This phase builds on the baseline models from Week 4 to tune, compare, and export a final production-ready model.

## Objective

To optimize the hyperparameters of the baseline models, analyze prediction errors, compare optimized models against each other, and select and export the best-performing final model for deployment.

## Repository Contents

| File | Description |
|---|---|
| `Optimization_Notebook.ipynb` | Main Jupyter/Colab notebook containing the full model optimization and selection workflow |
| `Model_Optimization_Report.pdf` | Written report summarizing the optimization process and results |
| `Final Trained Model/final_telco_churn_model.joblib` | Final trained model, exported with Joblib for deployment |
| `Final Trained Model/telco_feature_columns.joblib` | Feature column list required to align input data with the trained model |

## Workflow

The notebook follows these steps:

1. **GridSearchCV** — hyperparameter tuning of the baseline Logistic Regression model (regularization strength `C`, solver, class weighting)
2. **Best Hyperparameters** — identify the optimal parameter combination
3. **Evaluate Optimized Model**
4. **Before vs After Optimization** — compare baseline and optimized model performance
5. **Confusion Matrix — Before Optimization**
6. **Confusion Matrix — After Optimization**
7. **Classification Report**
8. **Error Analysis**
9. **Extract False Positive and False Negative Records**
10. **RandomizedSearchCV Experiment** — hyperparameter tuning of Random Forest as an alternative model
11. **Final Model Comparison** — compare optimized Logistic Regression vs. optimized Random Forest
12. **Final Model Selection** — select the best model based on overall performance
13. **Train Final Model**
14. **Export Final Model** — save with Joblib
15. **Load the Saved Model** — verify the exported model loads correctly
16. **Cross-Validation of Optimized Final Model**
17. **Open Joblib File in Google Colab**
18. **Final Conclusion**

## Results

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline, Week 4) | 80.55% | 55.88% | 0.6040 | 0.8420 |
| Logistic Regression (optimized, GridSearchCV) | — | 78.34% | 0.6175 | — |
| **Random Forest (optimized, RandomizedSearchCV) — Final Model** | ~77.00% | 72.73% | **0.6267** | **0.8431** |

Logistic Regression, the best baseline model from Week 4, was tuned with `GridSearchCV`, improving recall substantially (55.88% → 78.34%) at some cost to overall accuracy. A Random Forest model was separately optimized with `RandomizedSearchCV`, achieving the highest F1 Score (0.6267) and ROC-AUC (0.8431) with strong recall (72.73%).

**The optimized Random Forest was selected as the final model**, based on:
1. Highest F1 Score
2. Highest ROC-AUC
3. Strong recall — correctly identifies a large proportion of actual churners
4. A better balance between precision and recall than the more aggressively recall-oriented Logistic Regression
5. Ability to capture non-linear relationships between customer characteristics and churn

The final model was exported as `final_telco_churn_model.joblib` for future deployment.

## Key Libraries Used

- `pandas`
- `numpy`
- `scikit-learn` (`GridSearchCV`, `RandomizedSearchCV`, `RandomForestClassifier`, `LogisticRegression`, cross-validation, metrics)
- `joblib`
- `matplotlib`
- `seaborn`

## How to Run

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn joblib matplotlib seaborn
   ```
3. Open `Optimization_Notebook.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run the cells in order.

### Loading the Final Model

```python
import joblib

model = joblib.load("Final Trained Model/final_telco_churn_model.joblib")
feature_columns = joblib.load("Final Trained Model/telco_feature_columns.joblib")

# Ensure new input data is aligned to `feature_columns` before calling model.predict()
```

## Author

Amna Ali
