import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide", page_icon="📊")

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

st.markdown("""
<div class="main-header">
    <h1 class="main-title">📊 Exploratory Data Analysis</h1>
    <p class="main-subtitle">Analisis Mendalam Data Restoran Semarang</p>
</div>
""", unsafe_allow_html=True)

st.title("📊 Analisis Eksplorasi Data (EDA)")

df = load_data()

st.subheader("Filter Data")
col1, col2 = st.columns(2)
with col1:
    valid_resto_types = [x for x in df['resto_type'].unique() if pd.notna(x)]
    selected_resto_type = st.selectbox("Pilih Jenis Restoran", options=valid_resto_types, index=0)
with col2:
    rating_range = st.slider("Rentang Rating", min_value=0.0, max_value=5.0, value=(0.0, 5.0))
filtered_df = df[(df['resto_type'] == selected_resto_type) & (df['resto_rating'].between(rating_range[0], rating_range[1]))]

st.subheader("Ikhtisar Dataset")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Jumlah Baris**: {filtered_df.shape[0]}")
    st.write(f"**Jumlah Kolom**: {filtered_df.shape[1]}")
with col2:
    st.write("**Tipe Data**:")
    st.write(filtered_df.dtypes)

with st.expander("Lihat Sampel Dataset"):
    st.dataframe(filtered_df.head(), use_container_width=True)

st.subheader("Statistik Deskriptif")
st.write(filtered_df[['resto_rating', 'average_operation_hours']].describe())

st.subheader("Unduh Data")
if st.button("Unduh Dataset yang Difilter"):
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Unduh CSV",
        data=csv,
        file_name="filtered_resto_data.csv",
        mime="text/csv",
        use_container_width=True
    )

st.subheader("Visualisasi")

st.write("**Distribusi Rating Restoran**")
fig = px.histogram(filtered_df, x='resto_rating', nbins=20, title=f"Distribusi Rating Restoran ({selected_resto_type})")
fig.update_layout(xaxis_title="Rating", yaxis_title="Jumlah")
st.plotly_chart(fig, use_container_width=True)

st.write("**Jumlah Restoran untuk Jenis yang Dipilih**")
fig = px.histogram(filtered_df, x='resto_type', title=f"Jumlah Restoran ({selected_resto_type})")
fig.update_layout(xaxis_title="Jenis Restoran", yaxis_title="Jumlah", xaxis_tickangle=45)
st.plotly_chart(fig, use_container_width=True)

st.write("**Distribusi Rating untuk Jenis yang Dipilih**")
fig = px.box(filtered_df, x='resto_type', y='resto_rating', title=f"Rating Restoran ({selected_resto_type})")
fig.update_layout(xaxis_title="Jenis Restoran", yaxis_title="Rating", xaxis_tickangle=45)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Analisis Korelasi")
corr = filtered_df[['resto_rating', 'average_operation_hours', 'sell_halal_food', 'wifi_facility']].corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap Korelasi")
st.plotly_chart(fig, use_container_width=True)
