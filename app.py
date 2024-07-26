import streamlit as st
import pandas as pd
import re
import string

from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

st.set_page_config(
    page_icon="🎭", page_title="KEPUASAN PELANGGAN GOJEK", initial_sidebar_state="auto"
)

# https://blog.gojek.io/content/images/size/w2000/2021/02/API-Design-blog-01-01.png
page_bg_img = f"""
<style>

[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://blog.gojek.io/content/images/size/w2000/2021/02/API-Design-blog-01-01.png");
background-size: 100%;
background-height: auto;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

hide_menu_style = """
        <style>
        footer {visibility: visible;}
        footer:after{content:'Copyright @ 2023 Dwi Retno Irianti. All Rights Reserved'; display:block; position:relative; color:white}
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

m = """
<style>

.stAlert {
    text-color: #fff;
}
div.stButton > button:first-child {
    background-color: #000;
    border-radius:20px 20px 20px 20px;
}
div.stButton > button:hover {
    background-color: #004758;
    border: 0.5px solid #000;
    }
div.css-1om1ktf.e1y61itm0 {

        }
        textarea.st-cl {

          background-color: #ecefec;
          color: #000;
          font-family:"Roboto", serif;
          font-size: 16px;
        }
div.css-1dj3z61.e1iq63gx0 {
color: #fff;
}
p{
color: white;
}
</style>"""
st.markdown(m, unsafe_allow_html=True)

# from streamlit_extras.badges import badge

# logo_url = ("https://1000marcas.net/wp-content/uploads/2021/06/Gojek-Logo-2048x1280.png")


# badge(type="github", name="gojek", url="https://play.google.com/store/apps/details?id=com.gojek.app")

st.title("Klasifikasi Kepuasan Pelanggan Ojek Online (GOJEK)")

colored_header(
    label=" ",
    description="",
    color_name="green-70",
)

st.write(
    "IMPLEMENTASI TEXT PADA KLASIFIKASI KEPUASAN PELANGGAN TRANSPORTASI ONLINE (OJEK ONLINE) MENGGUNAKAN METODE KNN"
)

# Load the data
df = pd.read_csv("Data Ulasan Gojek.csv")

df.rename(
    columns={
        "userName": "Nama Pengguna",
        "score": "rating",
        "at": "tanggal",
        "content": "ulasan",
    },
    inplace=True,
)


# mengganti label pada kolom rating
def replace_rating(x):
    if x <= 3:
        return "Tidak Puas"
    else:
        return "Puas"


df["rating"] = df["rating"].apply(replace_rating)


# Preprocessing functions
def preprocess_text(df):
    text = df.lower()
    # Menghapus emotikon
    emoticon_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emotikon wajah
        "\U0001F300-\U0001F5FF"  # emotikon kategori simbol
        "\U0001F680-\U0001F6FF"  # emotikon kategori transportasi dan simbol bisnis
        "\U0001F1E0-\U0001F1FF"  # emotikon kategori bendera negara
        "\U00002702-\U000027B0"  # emotikon kategori tanda baca
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoticon_pattern.sub(r"", text)
    text = re.sub("[%s]" % re.escape(string.punctuation + string.digits), "", text)
    text = text.split()
    stop_words = set(stopwords.words("indonesian"))
    text = [word for word in text if word not in stop_words]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text


# Preprocess the reviews
df["proses ulasan"] = df["ulasan"].apply(preprocess_text)


# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["proses ulasan"])
Y = df["rating"]

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# tetangga_terdekat = st.slider('K ', value=440, max_value=800, min_value=1, help="k adalah tetangga terdekat")

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=440)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Daftar kata-kata positif dan negatif
kata_positif = [
    "senang",
    "membantu",
    "lancar",
    "mantap",
    "semoga",
    "bagus",
    "memuaskan",
    "memudahkan",
    "sangat puas",
    "bagus",
    "keren",
    "puas",
]
kata_negatif = [
    "kecewa",
    "buruk",
    "jelek",
    "mahal",
    "tidak",
    "error",
    "bajingan",
    "bangsat",
]


# Fungsi untuk mengklasifikasikan berdasarkan kata-kata positif dan negatif
def predict_satisfaction(review):
    processed_review = preprocess_text(review)

    # Mengubah ulasan menjadi fitur TF-IDF
    review_tfidf = tfidf.transform([processed_review])

    # Melakukan prediksi menggunakan model KNN
    prediction = knn.predict(review_tfidf)

    return prediction[0]

    # Inisialisasi bobot kata positif dan negatif
    bobot_positif = 1
    bobot_negatif = 1

    # Menghitung jumlah kata positif dan negatif dalam ulasan
    jumlah_positif = sum(
        bobot_positif for kata in kata_positif if kata in processed_review
    )
    jumlah_negatif = sum(
        bobot_negatif for kata in kata_negatif if kata in processed_review
    )

    # Klasifikasi berdasarkan jumlah kata positif dan negatif

    # # Menghitung skor sentimen berdasarkan jumlah terbobot kata-kata
    # skor_sentimen = jumlah_positif + jumlah_negatif
    #
    # # Klasifikasi berdasarkan ambang batas skor sentimen
    # # if skor_sentimen > 0:
    # #     return 'Puas'
    # # else:
    # #     return 'Tidak Puas'

    if jumlah_positif > jumlah_negatif:
        return "Puas"
    else:
        return "Tidak Puas"


# User input
user_input = st.text_area("Masukkan ulasan Anda ⬇")

if st.button("Klasifikasi"):
    if user_input:
        with st.spinner("Lagi Memuat Mohon Bersabar..."):
            time.sleep(5)
        # Make prediction using KNN model
        prediction = predict_satisfaction(user_input)

        # Display the predicted label
        if prediction == "Puas":
            st.success("Berdasarkan ulasan yang Anda berikan, Anda merasa PUAS.")
            st.balloons()
        else:
            st.warning("Berdasarkan ulasan yang Anda berikan, Anda merasa TIDAK PUAS.")
        # Display accuracy
        st.markdown(f"Nilai Akurasi : {accuracy:.2f}%")
        rain(
            emoji="🎭",
            font_size=30,
            falling_speed=10,
            animation_length="infinite",
        )
    else:
        st.info("Masukkan ulasan Anda sebelum melakukan klasifikasi")
        st.stop()
