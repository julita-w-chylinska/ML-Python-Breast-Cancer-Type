# Breast Tumor Type Prediction Model (ML in Python)

## Project Overview

This project demonstrates the process of building a machine learning model that predicts whether a breast tumor is **benign (non-cancerous)** or **malignant (cancerous)**. The project goal was to build a model achieving at least **0.95 accuracy**. The workflow includes minimal data cleaning, feature selection based on correlations, model training, and evaluation with ROC curve analysis.

> **Terminology note:** The dataset uses the word “cancer”, but the labels actually represent **tumor** diagnosis: malignant vs benign. In medical terminology, *cancer* typically refers to malignant neoplasms, while benign tumors are not considered cancer.

> **Disclaimer (educational use only):** This repository is an educational machine learning project based on a public, historical dataset. It is **not** a medical device and must not be used to diagnose, treat, or make clinical decisions without appropriate clinical validation, regulatory compliance, and oversight by qualified healthcare professionals. Any performance metrics reported here reflect this specific dataset and setup and may not generalize to real-world clinical populations.

## Dataset

The dataset is available in this repository (file: `Cancer_Data.csv`) and on Kaggle: https://www.kaggle.com/datasets/erdemtaha/cancer-data

## Project Structure and Results

### Import required libraries and objects

```Python
from  sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, roc_auc_score
```


### Load the dataset

```Python
os.chdir('../')
df = pd.read_csv('data/Cancer_Data.csv')
```

### Basic Data Exploration

Preview first rows:

```Python
df.head(30)
```

List all variables (column names), non-null counts, and data types:

```Python
df.info()
```

Check whether any missing values exist per variable (as a quick safety check):

```Python
df.isna().max()
```

Review the distribution of the `diagnosis` target labels:

```Python
df['diagnosis'].value_counts()
```

Compute basic descriptive statistics for numerical variables:

```Python
df.drop(columns=["id"]).describe()
```

<img width="1893" height="461" alt="image" src="https://github.com/user-attachments/assets/b42882c7-e8b9-41ff-aebd-42f8ec327aac" />


The dataset contains **569 records** and the following variables:
- `id` (dtype: `int64`)
- `diagnosis` (dtype: `object`) with values:
  - `M` = malignant
  - `B` = benign
- **30 numeric, continuous variables** (dtype: `float64`):  
  `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`, `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`, `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`
- A likely accidental column `Unnamed: 32` containing only missing values (`NaN`)


### Data Cleaning and Transformation

The data quality is reasonably high, so preparation includes only two steps:

1) Remove the last column (`Unnamed: 32`):

```Python
del df[df.columns[-1]]
```

2) Encode the target variable from categorical `diagnosis` to numeric `target`. Since the objective is to detect **malignant** tumors, malignant cases are encoded as `1`:

```Python
df['target'] = (df['diagnosis'] == 'M').astype(int)
```


### Correlation analysis (feature selection)

Display the full correlation matrix (all variables against all variables), using Spearman correlation:

```Python
plt.figure(figsize = (17,15))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2), annot = True, linewidths = 0.1)
plt.show()
```

<img width="1420" height="1360" alt="Correlation_Matrix_Cancer" src="https://github.com/user-attachments/assets/76b76a3b-f456-485d-bb79-baf471ef0824" />

The variable most strongly correlated with the target is `perimeter_worst`. It is included in the model first. Next, correlations of the remaining variables are evaluated both against the target and against `perimeter_worst`:

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst']],
            annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="665" alt="Correlation_Matrix_Cancer_A" src="https://github.com/user-attachments/assets/d99ea21b-46b4-42b6-9f38-6951e2926dd5" />

Selection rule used:
- include variables with **absolute correlation with the target ≥ 0.45**, and
- ensure correlation with already selected variables is **< 0.7** (to limit redundancy / multicollinearity).

Following this rule, the next selected variable is `perimeter_se`:

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se']], 
            annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="665" alt="Correlation_Matrix_Cancer_B" src="https://github.com/user-attachments/assets/78117d8a-dfa0-4564-9bb7-122b59942e54" />

The next variable meeting the criteria is `compactness_worst`, which is then added:

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se', 'compactness_worst']], annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="788" alt="Correlation_Matrix_Cancer_C" src="https://github.com/user-attachments/assets/60028dd7-6f2b-4de9-8b90-bf43b8697358" />

The next variable meeting the criteria is `concave points_se`, which is then added:

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se', 'compactness_worst', 'concave points_se']], annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="788" alt="Correlation_Matrix_Cancer_D" src="https://github.com/user-attachments/assets/d3ad24e8-f07f-42e3-9053-f3bc48b03df3" />

The next variable meeting the criteria is `texture_worst`, which is then added:

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se', 'compactness_worst', 'concave points_se', 'texture_worst']], annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="788" alt="Correlation_Matrix_Cancer_E" src="https://github.com/user-attachments/assets/119a7c57-bcb1-443b-b9fe-138de2604483" />

There is no more variables meeting the criteria.

### Final feature list used in the model

```Python
x_names = ['perimeter_worst', 'perimeter_se', 'compactness_worst', 'concave points_se', 'texture_worst']
```


### Outlier analysis

Outliers are identified using the **interquartile range (IQR)** method. To keep the modeling pipeline simple and robust, an observation will be excluded if it is flagged as an outlier for **any** of the selected model features. This is a deliberate trade-off: it reduces the training sample size, but may improve model stability by limiting the influence of extreme values.

Define a helper function that flags outliers:

```Python
def find_outliers(x, a = 1.5):
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    x_min = q1 - a * iqr
    x_max = q3 + a * iqr
    return (x < x_min) | (x > x_max)
```

Create additional columns indicating whether each observation is an outlier for each selected feature, and store those column names for later use:

```Python
outlier_column_names = []
for i in x_names:
    df[f'{i}_outlier'] = find_outliers(df[i])
    outlier_column_names.append(f'{i}_outlier')
```

As a result, the following outlier indicator columns are created (and stored in `outlier_column_names`):

<img width="416" height="190" alt="Zrzut ekranu 2026-02-09 161311" src="https://github.com/user-attachments/assets/093cf03d-e8cb-4c7f-8eea-ac5fc857162b" />

Create a single column indicating whether an observation is an outlier for **any** of the selected model features:

```Python
df['outlier_total'] = df[outlier_column_names].max(axis = 1)
```

Inspect how many observations are flagged as outliers (these will be excluded from model training and testing):

```Python
df['outlier_total'].value_counts()
```

<img width="201" height="113" alt="image" src="https://github.com/user-attachments/assets/942ff6af-dde0-4163-a214-c28a953ca186" />


### Train/test split

Import the train/test split function:

```Python
from sklearn.model_selection import train_test_split
```

Define `X` (input features) and `y` (target), excluding observations flagged as outliers:

```Python
X = df.loc[~(df.outlier_total), x_names]
y = df.loc[~(df.outlier_total), 'target']
```

Split into training and test sets:

```Python
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 123)
```

### Modeling

Create the model object:

```Python
model_1 = LogisticRegression()
```

Fit the model:

```Python
model_1.fit(train_x, train_y)
```


### Model evaluation

Generate predictions for both training and test sets:

```Python
train_pred = model_1.predict(train_x)
test_pred = model_1.predict(test_x)
```

Confusion matrices (training and test):

```Python
confusion_matrix(train_y, train_pred)
```

<img width="449" height="86" alt="image" src="https://github.com/user-attachments/assets/061e5966-a703-4191-a2a4-ea10b233f41d" />

```Python
confusion_matrix(test_y, test_pred)
```

<img width="420" height="96" alt="image" src="https://github.com/user-attachments/assets/1dc35f81-87da-406f-8d34-562f3ab49ad6" />

The confusion matrices look strong. Next, compute key metrics (Accuracy Score, TPR, FNR, TNR, and FPR) on training and test sets:

```Python
accuracy_score(train_y, train_pred)
```

Training accuracy is approximately **0.943**.

```Python
accuracy_score(test_y, test_pred)
```

Test accuracy is approximately **0.961**.

```Python
recall_score(test_y, test_pred)
```

Test **TPR** (sensitivity / recall) is approximately **0.946**.

```Python
1 - recall_score(test_y, test_pred)
```

Test **FNR** is approximately **0.054**.

```Python
recall_score(test_y, test_pred, pos_label=0)
```

Test **TNR** (specificity) is approximately **0.969**.

```Python
1 - recall_score(test_y, test_pred, pos_label=0)
```

Test **FPR** is approximately **0.031**.


### Interpretation of the current model performance

The model quality is high and appears stable (there is no clear sign of overfitting, as test performance does not degrade compared to training). Among malignant cases, approximately **95%** would be correctly predicted as malignant. Among benign cases, approximately **97%** would be correctly predicted as benign.

A key next step is aligning the model threshold with the “business” (clinical) objective:  
- prioritize detecting **all** malignant tumors (minimize false negatives), even if it increases false positives, or  
- prioritize minimizing false positives, even if it increases false negatives.


### ROC curve and decision threshold analysis

Compute predicted probabilities (for both training and test sets) of belonging to class `1` (malignant):

```Python
train_pred_p = model_1.predict_proba(train_x)[:,1]
test_pred_p = model_1.predict_proba(test_x)[:,1]
```

Compute FPR, TPR, and thresholds for the ROC curve:

```Python
fpr_train, tpr_train, threshold_train = roc_curve(train_y, train_pred_p)
fpr_test, tpr_test, threshold_test = roc_curve(test_y, test_pred_p)
```

Count the AUC score:

```Python
auc_train = round(roc_auc_score(train_y, train_pred_p), 3)
auc_test = round(roc_auc_score(test_y, test_pred_p), 3)
```

Plot the ROC curve:

```Python
plt.plot(fpr_train, tpr_train, label = "train")
plt.plot(fpr_test, tpr_test, label = "test")
plt.plot(np.arange(0,1,0.01), np.arange(0,1,0.01), '--')
plt.legend()
plt.annotate(f'AUC train: {auc_train}', xy = [0.2, 0.8])
plt.annotate(f'AUC test: {auc_test}', xy = [0.2, 0.75])

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

plt.title(f'Krzywa ROC')
plt.show()
```

<img width="567" height="455" alt="ROC_Curve_Cancer" src="https://github.com/user-attachments/assets/b678ca59-12fd-4e33-8960-ea6d3a0bb848" />

The ROC curve shape and AUC values confirm strong model performance.

If the priority is to detect **all** malignant tumors (i.e., ensure `TPR = 1`), we must select a threshold that achieves `TPR = 1` with the smallest possible `FPR`.

The following code finds the first index at which `tpr_test` equals exactly `1.0`, and then retrieves the corresponding threshold and FPR:

```Python
idxs = np.where(tpr_test == 1.0)[0]

if idxs.size == 0:
    raise ValueError("TPR never achieves 1.0 value at given threshold.")

idx_first_1 = idxs[0]

threshold_at_first_1 = threshold_test[idx_first_1]
fpr_at_first_1 = fpr_test[idx_first_1]

idx_first_1, threshold_at_first_1, fpr_at_first_1
```

<img width="458" height="50" alt="image" src="https://github.com/user-attachments/assets/56e30263-00ef-4672-8957-d4fde4d9494b" />


### Conclusions and recommendations

A threshold that ensures all malignant tumors are detected (i.e., **TPR = 1**) is **0.209611944215763**. However, at this threshold the **FPR** is **0.09375**, meaning that approximately **9%** of benign tumors would be incorrectly classified as malignant. Under the default threshold, only about **3%** of benign tumors are incorrectly classified as malignant.


## Author: Julita Wawreszuk-Chylińska

**LinkedIn**: [Julita Wawreszuk-Chylińska](https://www.linkedin.com/in/julita-wawreszuk-chylińska/)

Thank you for your interest in this project!
