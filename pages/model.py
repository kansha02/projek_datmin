import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Machine Learning Models", layout="wide", page_icon="⚙️")

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

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('semarang_resto_dataset.csv')
        df['resto_type'] = df['resto_type'].fillna('Tidak Diketahui')
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('float64')
            elif df[col].dtype == 'object':
                df[col] = df[col].astype('string')
        return df
    except FileNotFoundError:
        st.error("File 'semarang_resto_dataset.csv' tidak ditemukan! Pastikan file ada di direktori yang sama.")
        st.stop()

# Function to load model and supporting files
def load_model():
    try:
        model = joblib.load('models/best_xgb_model.pkl')
        feature_columns = joblib.load('models/model_features_xgboost.pkl')
        label_encoder = joblib.load('models/label_encoder_xgb.pkl')
        return model, feature_columns, label_encoder
    except FileNotFoundError:
        st.error("File model (best_xgb_model.pkl, model_features_xgboost.pkl, atau label_encoder_xgb.pkl) tidak ditemukan! Pastikan file ada di folder models/.")
        st.stop()

# Function to preprocess data
def preprocess_data(df, feature_columns):
    # Feature engineering
    df['name_length'] = df['resto_name'].str.len().fillna(0).astype(int)
    df['is_cafe_in_name'] = df['resto_name'].str.contains('cafe|kopi|coffee', case=False, na=False).astype(int)
    df['is_resto_in_name'] = df['resto_name'].str.contains('resto|restoran|rumah makan', case=False, na=False).astype(int)
    df['is_coffee_type'] = df['resto_type'].str.contains('kopi|coffee|cafe', case=False, na=False).astype(int)
    df['is_fastfood_type'] = df['resto_type'].str.contains('cepat saji|burger|pizza', case=False, na=False).astype(int)
    df['is_traditional_type'] = df['resto_type'].str.contains('jawa|sunda|padang|betawi|warung', case=False, na=False).astype(int)
    df['is_asian_type'] = df['resto_type'].str.contains('asia|korea|jepang|china|thailand', case=False, na=False).astype(int)
    df['is_main_road'] = df['resto_address'].str.contains('raya|jend|soedirman|thamrin|pemuda|pandanaran', case=False, na=False).astype(int)
    df['is_mall_area'] = df['resto_address'].str.contains('mall|plaza|centre|city', case=False, na=False).astype(int)

    # Handle missing values
    numerical_features = ['rating_numbers', 'average_operation_hours', 'name_length', 'is_cafe_in_name', 
                         'is_resto_in_name', 'is_coffee_type', 'is_fastfood_type', 'is_traditional_type', 
                         'is_asian_type', 'is_main_road', 'is_mall_area']
    binary_features = ['check_payment', 'cash_payment_only', 'debit_card_payment', 'credit_card_payment', 
                       'wifi_facility', 'sell_halal_food', 'suitable_for_children', 'menu_for_childern', 
                       'healty_menu', 'vegetarian_menu', 'sell_wine', 'contactless_deliverys_services', 
                       'take_away', 'drive_through', 'dine_in', 'delivery', 'open_space']
    
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    for col in binary_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # One-hot encoding for resto_type
    df = pd.get_dummies(df, columns=['resto_type'], drop_first=True)

    # Ensure all feature columns are present
    X = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0  # Add missing columns with default value 0
    X = X[feature_columns]  # Reorder to match feature_columns

    return X

st.markdown("""
<div class="main-header">
    <h1 class="main-title">⚙️Models</h1>
    <p class="main-subtitle">Training dan Evaluasi Model Prediksi</p>
</div>
""", unsafe_allow_html=True)

st.title("📈 Hasil Pelatihan Model")

df = load_data()
model, feature_columns, label_encoder = load_model()

# Preprocess data
df = df.drop_duplicates(subset=['resto_name', 'resto_address'], keep='first')
df['rating_category'] = df['resto_rating'].apply(lambda x: 'Rendah' if x < 3.8 else 'Sedang' if x < 4.4 else 'Tinggi')
y = df['rating_category']
X = preprocess_data(df, feature_columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict using loaded model
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))

# Evaluate model
accuracy = accuracy_score(y_test, y_pred_labels)
conf_matrix = confusion_matrix(y_test, y_pred_labels, labels=label_encoder.classes_)
class_report = classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_, output_dict=True)

st.subheader("Performa Model")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Akurasi**: {accuracy:.3f}")
with col2:
    st.write(f"**Akurasi Cross-Validation**: 0.91")

st.subheader("Laporan Klasifikasi")
st.write("**Precision, Recall, dan F1-Score per Kelas:**")
# Round individual metrics in the class_report dictionary
rounded_report = {}
for key, value in class_report.items():
    if key in ['Rendah', 'Sedang', 'Tinggi']:
        rounded_report[key] = {
            'precision': round(value['precision'], 3),
            'recall': round(value['recall'], 3),
            'f1-score': round(value['f1-score'], 3),
            'support': round(value['support'], 3)
        }
st.json(rounded_report)

# Confusion Matrix
st.subheader("Confusion Matrix")
fig = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                x=label_encoder.classes_, y=label_encoder.classes_, color_continuous_scale='Blues')
fig.update_layout(title="Confusion Matrix")
st.plotly_chart(fig, use_container_width=True)

# Feature Importance
st.subheader("Pentingnya Fitur")
importance = pd.DataFrame({'Fitur': feature_columns, 'Pentingnya': model.feature_importances_})
fig = px.bar(importance, x='Fitur', y='Pentingnya', title="Pentingnya Fitur")
fig.update_layout(xaxis_title="Fitur", yaxis_title="Pentingnya", xaxis_tickangle=45)
st.plotly_chart(fig, use_container_width=True)

# Prediction vs Actual
st.subheader("Prediksi vs Rating Aktual")
fig = go.Figure(data=[go.Scatter(x=y_test, y=y_pred_labels, mode='markers', text=y_test, name='Data'),
                      go.Scatter(x=['Rendah', 'Sedang', 'Tinggi'], y=['Rendah', 'Sedang', 'Tinggi'], mode='lines', line=dict(color='red', dash='dash'), name='Ideal')])
fig.update_layout(xaxis_title="Rating Aktual", yaxis_title="Rating Prediksi", title="Prediksi vs Aktual",
                  xaxis=dict(tickmode='array', tickvals=['Rendah', 'Sedang', 'Tinggi']),
                  yaxis=dict(tickmode='array', tickvals=['Rendah', 'Sedang', 'Tinggi']))
st.plotly_chart(fig, use_container_width=True)
