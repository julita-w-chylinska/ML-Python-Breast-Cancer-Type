# ML-Python-breast-cancer-type

This project demonstrates the [basic] proccess of building ML model for predicting whether the breast tumor* is benign (noncancerous) or malignant (cancerous). The goal was to built a model with at leart 0.95 accuracy score. This solution should help doctors make faster diagnoses and, therefore, reduce mortality. The process involved data preprocessing, chossing valid variables based on correlations, training a model and evaluating it.

*the data set says "cancer", but – in fact – cancer can't be benign, it's always malignant, so actually there is differentiation between types of tumors

## Import potrzebnych bibliotek i obiektów

```Python

from  sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

```

## Załadowanie zbioru danych

```Python

os.getcwd()
df = pd.read_csv('data/Cancer_Data.csv')

```

Zbiór
