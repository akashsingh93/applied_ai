# applied_ai

A collection of classical machine learning and NLP coursework notebooks, featuring exploratory data analysis, dimensionality reduction, and classic algorithms applied to real-world datasets from Kaggle and appliedaicourse.com.

## Contents

### Amazon Fine Food Reviews
- `02_Amazon_Fine_Food_Reviews_Analysis_TSNE.ipynb` — EDA + t-SNE visualization
- `11_Amazon_Fine_Food_Reviews_Analysis_Truncated_SVD.ipynb` — EDA + TruncatedSVD dimensionality reduction

### DonorsChoose (Project Funding Approval)
A numbered pipeline applying progressively sophisticated algorithms to predict funding approval:
- `2_DonorsChoose_EDA_TSNE-Copy1.ipynb` — EDA and t-SNE exploration
- `3_DonorsChoose_KNN_*.ipynb` — k-Nearest Neighbors (v0.2 and experimental versions)
- `4_DonorsChoose_NB_v0.2.ipynb` — Naive Bayes
- `5_DonorsChoose_LR_v0.2.ipynb` — Logistic Regression
- `7_DonorsChoose_SVM_Colab...pdf` — Support Vector Machines (exported PDF)
- `8_DonorsChoose_DT.ipynb` — Decision Trees
- `9_DonorsChoose_RF_GBDT_v*.ipynb` — Random Forest and Gradient Boosting (multiple versions)

### Haberman's Survival Dataset
- `Haberman_EDA.ipynb` — Exploratory analysis of patient survival data

### Linear Regression & Custom SGD
- `LinearRegression_Assignment.ipynb` — SGD-based linear regression from appliedaicourse.com
- `customSGD.py` — Hand-rolled stochastic gradient descent implementation

### Other Datasets
- `PersonalizedCancerDiagnosis_v0.1.ipynb` — Kaggle "Personalized Medicine" multi-class classification
- `Quora_Model_v0.1.ipynb` — Quora Question Pairs duplicate-detection modeling
- `apd_EDA.ipynb` + `adp_Modelling.ipynb` — KDD Cup 2012 Track 2 (ad click prediction) EDA and modeling

## Setup

This repo requires standard Python ML/data science libraries. Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Notebooks download datasets from external sources (Kaggle, appliedaicourse.com) and expect local paths as specified in each notebook. Data files are not checked in.

## Upcoming: Experimental Statistics & A/B Testing

This repo will expand to include implementations of core experimentation concepts:

1. **Build 1** — Sample-size and power calculator in Python
   - Inputs: baseline rate, MDE, alpha, power
   - Outputs: n per arm, expected duration
   - CLI + unit tests

2. **Build 2** — A/B test analysis notebook (Cookie Cats dataset)
   - SRM check, primary metric with CI
   - Guardrails and segment cuts (FDR-corrected)
   - Executive summary

3. **Build 3** — Clustered experiment simulation
   - Show variance underestimation and false positive inflation when clustering is ignored
   - Implement delta method for ratio metrics; validate against bootstrap

4. **Build 4** — Sequential testing (mSPRT / always-valid p-values)
   - Compare naive daily peeking (~30% false positive rate) vs. corrected method (~5%)
   - Plot convergence on simulated streaming data

## License

Code in `classical_ml_nlp/` is licensed under the terms in `classical_ml_nlp/LICENSE`.