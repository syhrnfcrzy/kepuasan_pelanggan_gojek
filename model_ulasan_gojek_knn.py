# -*- coding: utf-8 -*-
"""Model Ulasan Gojek KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kydDmV2lpV_CWX3C-niVPdfU91UtvY6X

# IMPLEMENTASI TEXT PADA KLASIFIKASI KEPUASAN PELANGGAN TRANSPORTASI ONLINE (OJEK ONLINE) MENGGUNAKAN METODE KNN

### Import Library
"""

import pandas as pd
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')

"""### Preprocessing Data"""

# MEMASUKAN DATASET YANG DIDAPATKAN DARI HASIL SCRAPING PADA GOOGLE PLAY STORE
df = pd.read_csv("Data Ulasan Gojek.csv")

#MNEMAPILKAN 1000 DATA PERTAMA DARI DATASET
df

#MENGGANTI NAMA KOLOM PADA DATASET
df.rename(columns = {"userName": "Nama Pengguna", "score": "rating", "at": "tanggal", "content": "ulasan"}, inplace=True)

#MNEMAPILKAN 1000 DATA PERTAMA DARI DATASET YANG TELAH DIUBAH KOLOMNYA
df

df.drop('Nama Pengguna', axis=1, inplace=True)

# mengganti label pada kolom rating JIKA RATINGNYA KURANG DARI 3 ADALAH TIDAK PUAS DAN JIKA LEBIH DARI 3 (4 DAN 5) ADALAH PUAS
def replace_rating(x):
    if x <= 3:
        return 'Tidak Puas'
    else:
        return 'Puas'

df['rating'] = df['rating'].apply(replace_rating)

#MNEMAPILKAN 1000 DATA PERTAMA DARI DATASET YANG TELAH DILABELI PUAS DAN TIDAK PUAS
df

# Memindahkan kolom tanggal ke kiri dan kolom rating ke paling kanan
df = df[['tanggal', 'ulasan', 'rating']]
df

"""### Data Cleaning"""

# menghapus tanda baca dan angka
df['ulasan'] = df['ulasan'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation + string.digits), '', x))
df

"""### Case Folding"""

# case folding
df['ulasan'] = df['ulasan'].apply(lambda x: x.lower())
df

"""### Tokenisasi"""

# tokenisasi
df['ulasan'] = df['ulasan'].apply(lambda x: x.split())
df

"""### StopWord"""

# stopword removal
stop_words = set(stopwords.words('indonesian'))
df['ulasan'] = df['ulasan'].apply(lambda x: [word for word in x if word not in stop_words])
df

"""### Stemming"""

# stemming
stemmer = PorterStemmer()
df['ulasan'] = df['ulasan'].apply(lambda x: [stemmer.stem(word) for word in x])
df

# menggabungkan kata-kata kembali menjadi teks
df['ulasan'] = df['ulasan'].apply(lambda x: ' '.join(x))
df

"""### TF-IDF"""

# ekstraksi fitur menggunakan TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['ulasan'])
Y = df['rating']

"""### Pembagian Dataset"""

# pembagian data train dan test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

"""### Algoritma K-Nearest Neighbor"""

# pelatihan model KNN
knn = KNeighborsClassifier(n_neighbors=440)
knn.fit(X_train, y_train)

# prediksi menggunakan data test
y_pred = knn.predict(X_test)

"""### Nilai Akurasi"""

# evaluasi model
acc = accuracy_score(y_test, y_pred)
print('Akurasi model KNN: {:.2f}%'.format(acc*100))

# Jumlah data latih
jumlah_data_latih = X_train.shape[0]
print("Jumlah data latih:", jumlah_data_latih)

# Jumlah data uji
jumlah_data_uji = X_test.shape[0]
print("Jumlah data uji:", jumlah_data_uji)