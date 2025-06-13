import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import random

st.set_page_config(page_title="Rating Prediction", layout="wide", page_icon="⭐")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --primary-color: #2563eb;
        --primary-light: #3b82f6;
        --primary-lighter: #60a5fa;
        --primary-lightest: #dbeafe;
        --secondary-color: #1e40af;
        --secondary-light: #1d4ed8;
        --accent-color: #0ea5e9;
        --accent-light: #38bdf8;
        --neutral-50: #f8fafc;
        --neutral-100: #f1f5f9;
        --neutral-200: #e2e8f0;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1e293b;
        --success-color: #059669;
        --success-light: #10b981;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.15);
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3.2em;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-subtitle {
        font-size: 1.3em;
        margin-top: 12px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
    }
    
    .prediction-result {
        color: var(--primary-color);
        font-size: 1.2em;
        margin: 10px 0;
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2em;
        }
        
        .main-subtitle {
            font-size: 1.1em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to load model, label encoder, and dataset
def load_model_and_data():
    try:
        model = joblib.load('models/model.pkl')
        le = joblib.load('models/label_encoder.pkl')  # Use the old label encoder
        data = pd.read_csv('semarang_resto_dataset.csv')
        valid_resto_types = data['resto_type'].dropna().unique()
        return model, le, data, valid_resto_types
    except FileNotFoundError:
        st.error("File model atau dataset tidak ditemukan! Pastikan model.pkl, label_encoder.pkl, dan semarang_resto_dataset.csv ada di folder models/.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat model atau dataset: {str(e)}")
        st.stop()

# Function to preprocess input data for old model
def preprocess_input(resto_type, avg_hours, halal_food, wifi, toilet, children, dine_in, take_away, delivery, open_space, le):
    # Encode resto_type using label encoder
    encoded_resto_type = le.transform([resto_type])[0] if resto_type in le.classes_ else 0  # Default to 0 if not in classes
    
    input_data = pd.DataFrame({
        'average_operation_hours': [avg_hours],
        'sell_halal_food': [1 if halal_food else 0],
        'wifi_facility': [1 if wifi else 0],
        'toilet_facility': [1 if toilet else 0],
        'suitable_for_children': [1 if children else 0],
        'dine_in': [1 if dine_in else 0],
        'take_away': [1 if take_away else 0],
        'delivery': [1 if delivery else 0],
        'open_space': [1 if open_space else 0],
        'resto_type': [encoded_resto_type]  # Encoded value
    })
    return input_data

# Initialize session state
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

st.markdown("""
<div class="main-header">
    <h1 class="main-title">⭐ Rating Prediction</h1>
    <p class="main-subtitle">Prediksi Rating Restoran Baru</p>
</div>
""", unsafe_allow_html=True)

# Load model and data
model, le, data, valid_resto_types = load_model_and_data()

st.subheader("Masukkan Detail Restoran")
with st.form("formulir_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        valid_classes = valid_resto_types.tolist()  # Use valid_resto_types from dataset
        if not valid_classes:
            st.error("Tidak ada jenis restoran di dataset. Periksa semarang_resto_dataset.csv.")
            st.stop()
        resto_type = st.selectbox("Jenis Restoran", valid_classes, key="resto_type_select")
        avg_hours = st.number_input("Rata-rata Jam Operasional", min_value=0.0, max_value=24.0, value=10.0)
        halal_food = st.checkbox("Menyediakan Makanan Halal")
        wifi = st.checkbox("Fasilitas WiFi")
        toilet = st.checkbox("Fasilitas Toilet")
    with col2:
        children = st.checkbox("Cocok untuk Anak-anak")
        dine_in = st.checkbox("Tersedia Dine-In")
        take_away = st.checkbox("Tersedia Take-Away")
        delivery = st.checkbox("Tersedia Delivery")
        open_space = st.checkbox("Tersedia Ruang Terbuka")
    submitted = st.form_submit_button("Prediksi Rating")

    if submitted and avg_hours > 0:
        # Filter dataset berdasarkan resto_type yang dipilih
        filtered_data = data[data['resto_type'] == resto_type]
        if not filtered_data.empty:
            resto_name = random.choice(filtered_data['resto_name'].values)
        else:
            resto_name = f"New {resto_type.replace('_', ' ').title()}"

        # Preprocess input data with encoding
        input_data = preprocess_input(resto_type, avg_hours, halal_food, wifi, toilet, children, dine_in, take_away, delivery, open_space, le)

        try:
            prediction = model.predict(input_data)[0]
            st.subheader("Hasil Prediksi")
            st.markdown("""
                <div class="prediction-result">
                    <p>Nama Restoran: <strong>{}</strong></p>
                    <p>Jenis: <strong>{}</strong></p>
                    <p>Rating: <strong>{:.2f}/5 ⭐</strong></p>
                </div>
            """.format(resto_name, resto_type.replace('_', ' ').title(), prediction), unsafe_allow_html=True)

            # Pie chart for numerical prediction
            fig = px.pie(values=[prediction, 5-prediction], names=['Rating', 'Sisa'], hole=0.4,
                         title="Visualisasi Rating", color_discrete_sequence=['var(--primary-color)', 'var(--neutral-200)'])
            st.plotly_chart(fig, use_container_width=True)

            # Store prediction data
            st.session_state.prediction_data = pd.DataFrame({
                'Resto_Name': [resto_name],
                'Resto_Type': [resto_type],
                'Average_Operation_Hours': [avg_hours],
                'Predicted_Rating': [prediction]
            })
        except Exception as e:
            st.error(f"Terjadi error dalam prediksi: {str(e)}. Pastikan fitur dalam preprocess_input sesuai dengan model.pkl atau periksa encoding resto_type.")
    elif submitted and avg_hours <= 0:
        st.warning("Jam operasional harus lebih dari 0!")

# Display download button
if st.session_state.prediction_data is not None:
    csv = st.session_state.prediction_data.to_csv(index=False)
    st.download_button("Unduh Hasil Prediksi", csv, "prediction_result.csv", "text/csv", use_container_width=True)
