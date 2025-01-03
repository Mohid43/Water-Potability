# -*- coding: utf-8 -*-

import pandas as pd
file_path = '/content/drive/MyDrive/water_potability.csv'
water_data = pd.read_csv(file_path)
water_data.head()

"""**MENELAAH,VALIDASI dan VISUALISASI DATA**"""

water_data.info()

water_data.nunique()

"""**MENENTUKAN OBJEK DATA**"""

water_data.describe()

missing_values = water_data.isnull().sum()

water_data_imputed = water_data.fillna(water_data.mean())

Q1 = water_data_imputed.quantile(0.25)
Q3 = water_data_imputed.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((water_data_imputed < lower_bound) | (water_data_imputed > upper_bound))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

sns.barplot(data=water_data)
plt.title('Distribusi Data Kualitas Air Sebelum Imputasi')

plt.subplot(1, 2, 2)

sns.barplot(data=water_data_imputed)
plt.title('Distribusi Data Kualitas Air Setelah Imputasi')

plt.tight_layout()
plt.show()

"""**MEMBESIHKAN DATA**"""

plt.figure(figsize=(12, 8))
sns.heatmap(water_data_imputed.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasi Antar Fitur')
plt.show()

import matplotlib.pyplot as plt # Import matplotlib to use plotting functions
import numpy as np

data = np.random.randn(len(water_data))

# Membuat histogram
plt.hist(data, bins=30, edgecolor='black')

# Menambahkan judul dan label
plt.title('Histogram Distribusi Data')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')

# Menampilkan plot
plt.show()

import matplotlib.pyplot as plt # Import matplotlib to use plotting functions
import numpy as np
import seaborn as sns # Import seaborn for statistical data visualization

parameters = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
              'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Mengatur ukuran figure
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

# Membuat histogram untuk setiap parameter
# Use water_data_imputed instead of data
for i, param in enumerate(parameters):
    sns.histplot(water_data_imputed[param], bins=30, kde=True, ax=axes[i], color='blue')
    axes[i].set_title(f'Distribution of {param}')
    axes[i].set_xlabel(param)
    axes[i].set_ylabel('Frequency')

# Mengatur layout
plt.tight_layout()
plt.show()

"""**KONTRUKSI DATA**"""

import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/water_potability.csv')

# Mengecek tipe data
print(data.dtypes)
# Mengecek tipe data
print(data.dtypes)

"""**PERMODELAN**"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = water_data_imputed

X = data.drop('Potability', axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = water_data_imputed

X = data.drop('Potability', axis=1)
y = data['Potability']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Daftar model
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

# Membandingkan model
for name, model in models.items():
    # Melatih model
    model.fit(X_train, y_train)
    # Prediksi
    y_pred = model.predict(X_test)
    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

"""**EVALUASI**"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('/content/drive/MyDrive/water_potability.csv')

X = data.drop('Potability', axis=1)
y = data['Potability']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Daftar model
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

"""**DEPLOYMENT**"""

corr = water_data.corr()
target_corr = corr['Potability'].sort_values(ascending=False)
print("kolerasi fitur dengan Potability:\n", target_corr)

