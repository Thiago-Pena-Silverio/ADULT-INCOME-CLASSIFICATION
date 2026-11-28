# Adult Income Classification

Can a classifier identify people earning more than $50,000 a year in the UCI Adult dataset? This project compares logistic regression with two random forests and examines the trade-off between finding more high-income cases and making more false-positive predictions.

## 🎓 Context

Thiago Pena Silverio completed this project for the Applied Machine Learning module. The experiment uses ten demographic and employment features from the historical [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult). After removing 24 duplicate rows, the notebook splits the remaining records into 26,029 training cases and 6,508 test cases. Preprocessing and five-fold model selection use the training data.

## 📁 Folders

- [Notebooks](Notebooks/Adult_Income_Classification.ipynb) contains the completed analysis, including exploratory plots, a 12-configuration grid search, confusion matrices, ROC curves and feature importance.

## 🛠️ Tech Stack

Python, pandas, scikit-learn, Matplotlib, seaborn, joblib and Jupyter.

## 🎖️ Results

The positive class is annual income above $50,000. Scores below come from the held-out test set in the submitted notebook.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8563 | 0.7471 | 0.6103 | 0.6718 | 0.9076 |
| Initial random forest | 0.8479 | 0.7079 | 0.6276 | 0.6653 | 0.8921 |
| Tuned random forest | 0.8245 | 0.5942 | 0.8571 | 0.7018 | 0.9179 |

The tuned forest identified 1,344 of the 1,568 people in the higher-income group and produced 918 false positives. Logistic regression had the highest precision. These results describe the notebook's stratified split of the UCI training file.

A fresh end-to-end run with Python 3.13 and scikit-learn 1.6.1 reproduced the displayed scores for logistic regression and the initial forest to four decimal places. The tuned forest returned accuracy 0.8397, precision 0.6279, recall 0.8221, F1 0.7120 and ROC-AUC 0.9164. It still had the highest F1 and recall. The exact cause of the difference from the submitted run is unverified.

---

## 🖥️ How to Run

From the repository root, install the dependencies and open the notebook:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab Notebooks/Adult_Income_Classification.ipynb
```

Choose **Run All Cells**. The data-loading cell downloads the public UCI file and the final code cell saves the trained pipeline to `model_deployment/adult.pkl` under the kernel's working directory. The Flask demo mentioned in the assignment is separate from this notebook.
