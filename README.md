# ML-Python-breast-cancer-type

## Project Overview

This project demonstrates the [basic] proccess of building ML model for predicting whether the breast tumor* is benign (noncancerous) or malignant (cancerous). The goal was to built a model with at leart 0.95 accuracy score. This solution should help doctors make faster diagnoses and, therefore, reduce mortality. The process involved data preprocessing, chossing valid variables based on correlations, training a model and evaluating it.

*the data set says "cancer", but – in fact – cancer can't be benign, it's always malignant, so actually there is differentiation between types of tumors

## Project Structure and Results

# Import potrzebnych bibliotek i obiektów

```Python

from  sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

```

# Załadowanie zbioru danych

```Python

os.getcwd()
df = pd.read_csv('data/Cancer_Data.csv')

```

Zbiór jest dostępny w plikach tego repozytorium (plik o nazwie `Cancer_Data.csv`) oraz na stronie https://www.kaggle.com/datasets/erdemtaha/cancer-data

# Eksploracja danych

Podejrzenie pierwszych 30 rekordów
```Python
df.head(30)
```
Spis wszystkich zmiennych (nazw kolumn) wraz z liczebnością niepustych wartości i typem danych
```Python
df.info()
```
Sprawdzenie obecności braków danych dla każdej zmiennej (na wszelki wypadek, mimo że było to "ręcznie" widoczne w wyniku poprzedniego kodu)
```Python
df.isna().max()
```
Sprawdzenie wartości zmiennej `diagnosis` wraz z ich liczebnością
```Python
df['diagnosis'].value_counts()
```
Sprawdzenie podstawowych statystyk dla zmiennych numerycznych/liczbowych
```Python
df.describe()
```

Zbiór danych okazuje się mieć 569 rekordów oraz następujące zmienne:
* `id` o typie "int64"
* `diagnosis` o typie "object" z wartościami "M" (oznaczającą "malignant") i "B" (oznaczającą "benign")
* 29 zmiennych o typie "float64": `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`, `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`, `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`
* prawdopodobnie przypadkowo utworzona zmienna `Unnamed: 32` z samymi pustymi wartościami (NaN)



