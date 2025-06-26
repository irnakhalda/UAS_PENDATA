import streamlit as st
import pickle
import pandas as pd

# Load model dan label encoders
model = pickle.load(open("model_rf.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Daftar kolom input sesuai dataset
input_columns = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
    'Thyroid Function', 'Physical Examination', 'Adenopathy',
    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

# Judul aplikasi
st.title("Prediksi Kekambuhan Kanker Tiroid")

# Form input pengguna
st.subheader("Masukkan data pasien:")
user_input = {}

for col in input_columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        user_input[col] = st.selectbox(f"{col}:", options)
    else:
        user_input[col] = st.number_input(f"{col}:", step=1.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    input_data = []
    for col in input_columns:
        val = user_input[col]
        if col in label_encoders:
            val = label_encoders[col].transform([val])[0]
        else:
            val = float(val)
        input_data.append(val)

    input_df = pd.DataFrame([input_data], columns=input_columns)
    prediction = model.predict(input_df)[0]
    result = label_encoders['Recurred'].inverse_transform([prediction])[0]

    # Tampilkan hasil prediksi
    st.success(f"📋 Hasil Prediksi: {result}")
