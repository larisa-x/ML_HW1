import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


# EDA

TRAIN_DATA_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"

@st.cache_data
def load_train_data():
    return pd.read_csv(TRAIN_DATA_URL)

st.markdown("---")
st.header("📊 EDA на train-датасете")

try:
    df_train = load_train_data()

    st.subheader("Первые строки train")
    st.dataframe(df_train.head())

    st.subheader("Размер и пропуски")
    st.write("shape:", df_train.shape)
    st.write(df_train.isna().sum())

    st.subheader("Распределение года выпуска (year)")
    if "year" in df_train.columns:
        st.bar_chart(df_train["year"].value_counts().sort_index())

    st.subheader("Распределение пробега (km_driven) — histogram")
    km = pd.to_numeric(df_train["km_driven"], errors="coerce").dropna()
    km = km[km <= km.quantile(0.99)]

    fig, ax = plt.subplots()
    ax.hist(km, bins=30)
    ax.set_xlabel("km_driven")
    ax.set_ylabel("count")
    st.pyplot(fig)

    st.subheader("Распределение мощности (max_power) — histogram")
    mp = pd.to_numeric(df_train["max_power"], errors="coerce").dropna()
    mp = mp[mp <= mp.quantile(0.99)]  

    fig, ax = plt.subplots()
    ax.hist(mp, bins=30)
    ax.set_xlabel("max_power")
    ax.set_ylabel("count")
    st.pyplot(fig)



except Exception as e:
    st.warning(f"Не удалось загрузить train-датасет для EDA: {e}")


# предсказания

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction")
st.write("Загрузите CSV с признаками — приложение сделает предсказание цены")

print("📦 Загружаем модель и список признаков...")

model_dir = "models"
model_path = os.path.join(model_dir, "model.pkl")
feature_names_path = os.path.join(model_dir, "feature_names.pkl")

with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

with open(feature_names_path, "rb") as f:
    loaded_features = pickle.load(f)

st.success("✅ Модель загружена")
st.write(f"Количество признаков: {len(loaded_features)}")
st.write("Список признаков:")
st.write(loaded_features)

st.header("📄 Загрузка CSV")
uploaded_file = st.file_uploader(
    "Загрузите CSV-файл с признаками (без selling_price)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Первые строки загруженного файла:")
    st.dataframe(df.head())

    X = df[loaded_features]

    if st.button("🔮 Сделать предсказание"):
        preds = loaded_model.predict(X)

        df_out = df.copy()
        df_out["predicted_price"] = preds

        st.success("✅ Предсказания готовы")
        st.dataframe(df_out.head())


# коэффициенты модели
st.markdown("---")
st.header("📊 Коэффициенты линейной модели")

coefs = pd.DataFrame({
    "feature": loaded_features,
    "coefficient": loaded_model.coef_
}).sort_values(by="coefficient", key=abs, ascending=False)

st.dataframe(coefs)

st.bar_chart(coefs.set_index("feature"))
