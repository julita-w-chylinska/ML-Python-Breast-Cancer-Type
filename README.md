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
* 29 zmiennych numerycznych/liczbowych/ciągłych o typie "float64": `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`, `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`, `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`
* prawdopodobnie przypadkowo utworzona zmienna `Unnamed: 32` z samymi pustymi wartościami (NaN)

# Czyszczenie i przekształcanie danych

The data exhibited a reasonable/hight level of quality so data preparation involved only two actions:
1) Usunięcie ostatniej zmiennej (`Unnamed: 32`)
```Python
del df[df.columns[-1]]
```
2) Enkodowanie zmiennej celu z tekstowej/kategorycznej `diagnosis` na numeryczną `target` (chcemy modelować zjawisko wykrywania nowotworu złośliwego, więc jako 1 oznaczymy właśnie ten typ nowotworu)
```Python
df['target'] = (df['diagnosis'] == 'M').astype(int)
```

# Sprawdzanie korelacji między zmiennymi (w celu wybrania najważniejszych do modelu)

Wyświetlenie macierzy korelacji (wszystkich zmiennych ze wszystkimi zmiennymi)
```Python
plt.figure(figsize = (17,15))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2), annot = True, linewidths = 0.1)
plt.show()
```
<img width="1420" height="1360" alt="Correlation_Matrix_Cancer" src="https://github.com/user-attachments/assets/76b76a3b-f456-485d-bb79-baf471ef0824" />

Zmienną nasilniej skorelowaną ze zmienną celu jest `perimeter_worst`. Włączymy ją do modelu, a teraz sprawdzamy korelację reszty zmiennych zarówno ze zmienną celu, jak i przed chwilą wybraną zmienną `perimeter_worst`

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst']],
            annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="665" alt="Correlation_Matrix_Cancer_A" src="https://github.com/user-attachments/assets/d99ea21b-46b4-42b6-9f38-6951e2926dd5" />

Przyjmując zasadę o włączaniu do modelu takich zmiennych, których wartość bezwględna współczynnika korelacji ze zmienną celu wynosi CO NAJMNIEJ 0.45 (jako że wiele zmiennych ma dużo silniejszą korelację), a z innymi – włączonymi już do modelu – zmiennymi MNIEJ NIŻ 0.7, jako następną zmienną wybieramy `perimeter_se` i dołączamy ją do tabeli korelacji.

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se']], 
            annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="665" alt="Correlation_Matrix_Cancer_B" src="https://github.com/user-attachments/assets/78117d8a-dfa0-4564-9bb7-122b59942e54" />

Następną zmienną spełniającą warunki jest `compactness_mean`. Wybieramy ją o modelu i dołączamy do tabeli korelacji.

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se', 'compactness_worst']], annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="788" alt="Correlation_Matrix_Cancer_C" src="https://github.com/user-attachments/assets/60028dd7-6f2b-4de9-8b90-bf43b8697358" />

Następną zmienną spełniającą warunki jest `concave points_se`. Wybieramy ją o modelu i dołączamy do tabeli korelacji.

```Python
plt.figure(figsize = (6,8))
sns.heatmap(round(df[df.columns[2:]].corr('spearman').sort_values(by = 'target'), 2)[['target', 'perimeter_worst', 'perimeter_se', 'compactness_worst', 'concave points_se']], annot = True, linewidths = 0.1)
plt.show()
```

<img width="669" height="788" alt="Correlation_Matrix_Cancer_D" src="https://github.com/user-attachments/assets/d3ad24e8-f07f-42e3-9053-f3bc48b03df3" />

Następną zmienną spełniającą warunki jest `texture_worst`. Jest to też ostatnia zmienna, która spełniałaby ustalone warunki.

# Ostateczny wybór zmiennych do modelu i zapisanie ich na liście

```Python
x_names = ['perimeter_worst', 'perimeter_se', 'compactness_worst', 'concave points_se', 'texture_worst']
```

# Badanie outlierów

Do zbadania outlierów użyjemy metody z rozstępem międzykwartylowym [tu może jeszcze dopisać wyjaśnienie]

Zdefiniowanie funkcji identyfikującej outliery

```Python
def find_outliers(x, a = 1.5):
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    x_min = q1 - a * iqr
    x_max = q3 + a * iqr
    return (x < x_min) | (x > x_max)
```

Zbudowanie pętli tworzącej nowe kolumny zawierające informacje, czy w danym wierszu pojawił się outlier dla danej zmiennej (mającej wziąć udział w modelu), oraz dodającej nazwy tych kolumn do listy (będzie ona za chwilę potrzebna).

```Python
outlier_column_names = []
for i in x_names:
    df[f'{i}_outlier'] = find_outliers(df[i])
    outlier_column_names.append(f'{i}_outlier')
```

W wyniku zadziałania powyższej pętli, to zbioru danych zostały dołączone następujące kolumny/zmienne (których nazwy zostały mieszczone w liście `outlier_column_names`):

<img width="416" height="190" alt="Zrzut ekranu 2026-02-09 161311" src="https://github.com/user-attachments/assets/093cf03d-e8cb-4c7f-8eea-ac5fc857162b" />

Stworzenie nowej kolumny/zmiennej określającej, czy w danym wierszu/obserwacji/rekordzie pojawiła się wartość odstająca (outlier) dla **którejkolwiek** ze zmiennych (mających wziąć udział w modelu).

```Python
df['outlier_total'] = df[outlier_column_names].max(axis = 1)
```

Zorientowanie się w ilości obserwacji/rekordów, które zostają uznane w całości jako outliery (nie będą wzięte pod uwagę w trenowaniu i testowaniu modelu).

```Python
df['outlier_total'].value_counts()
```

# Podział zbioru na treningowe i testowe

Import obiektu potrzebnego do wykonania podziału

```Python
from sklearn.model_selection import train_test_split
```

Zdefiniowanie zbioru X (ze zmiennymi wejściowymi/objaśniającymi) i zbioru y (ze zmienną wyjściową/objaśnianą/celu)

```Python
X = df.loc[~(df.outlier_total), x_names]
y = df.loc[~(df.outlier_total), 'target']
```

Podział zbiorów X i y na podzbiory treningowe i testowe (z określeniem ich proporcji)

```Python
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 123)
```

# Modelowanie / Tworzenie modelu

Stworzenie obiektu modelu

```Python
model_1 = LogisticRegression()
```

Estymacja modelu

```Python
model_1.fit(train_x, train_y)
```

# Ocena jakości modelu

Wywołanie / stworzenie predykcji na zbiorze treningowym i testowym

```Python
train_pred = model_1.predict(train_x)
test_pred = model_1.predict(test_x)
```




