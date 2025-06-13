import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Dashboard Restoran Semarang",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Color Variables */
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
    
    /* Header */
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
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.12);
    }
    
    .info-card h3 {
        font-size: 1.4em;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .info-card p {
        margin: 8px 0;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .info-card .highlight {
        color: var(--primary-lightest);
        font-weight: 600;
    }
    
    /* Stats Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .stat-card:nth-child(1) {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
    }
    
    .stat-card:nth-child(2) {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
    
    .stat-card:nth-child(3) {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--primary-lighter) 100%);
    }
    
    .stat-card:nth-child(4) {
        background: linear-gradient(135deg, var(--success-color) 0%, var(--success-light) 100%);
    }
    
    .stat-number {
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1em;
        font-weight: 500;
        opacity: 0.95;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.06);
        border: 1px solid var(--neutral-200);
        border-top: 4px solid var(--primary-color);
    }
    
    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.4em;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .feature-description {
        color: var(--neutral-600);
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Technology Stack with Better Animations */
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 2rem 0;
        justify-content: center;
    }
    
    .tech-item {
        color: white;
        padding: 0.7rem 1.3rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        position: relative;
        transition: all 0.3s ease;
        opacity: 1;
        transform: translateY(0);
        animation: fadeInScale 0.6s ease-out;
    }
    
    /* Background Colors with Animation Delays */
    .tech-item:nth-child(1) { 
        background: var(--primary-color);
        animation-delay: 0.1s;
    }
    .tech-item:nth-child(2) { 
        background: var(--primary-light);
        animation-delay: 0.15s;
    }
    .tech-item:nth-child(3) { 
        background: var(--secondary-color);
        animation-delay: 0.2s;
    }
    .tech-item:nth-child(4) { 
        background: var(--accent-color);
        animation-delay: 0.25s;
    }
    .tech-item:nth-child(5) { 
        background: var(--primary-lighter);
        animation-delay: 0.3s;
    }
    .tech-item:nth-child(6) { 
        background: var(--secondary-light);
        animation-delay: 0.35s;
    }
    
    /* Simple Hover Effects */
    .tech-item:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        filter: brightness(1.1);
    }
    
    /* Floating Animation for Even Items */
    .tech-item:nth-child(even) {
        animation: fadeInScale 0.6s ease-out, float 3s ease-in-out infinite;
    }
    
    .tech-item:nth-child(even) {
        animation-delay: 0.1s, 1s;
    }
    
    /* Glow Effect for Odd Items */
    .tech-item:nth-child(odd):hover {
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
    }
    
    /* Keyframe Animations */
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-3px);
        }
    }
    
    /* Gentle Pulse for Active Tech */
    @keyframes gentlePulse {
        0%, 100% {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        50% {
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
        }
    }
    
    /* Active State Animation */
    .tech-item:active {
        transform: translateY(-2px) scale(0.98);
        transition: all 0.1s ease;
    }
    
    /* Two Column Layout */
    .two-column {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    /* Welcome Section */
    .welcome-section {
        background: var(--neutral-50);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid var(--neutral-200);
        border-left: 6px solid var(--primary-color);
        margin-bottom: 2rem;
    }
    
    .welcome-section h3 {
        color: var(--primary-color);
        font-size: 1.5em;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .welcome-section p {
        color: var(--neutral-700);
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .welcome-section .highlight {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2em;
        }
        
        .main-subtitle {
            font-size: 1.1em;
        }
        
        .stats-container {
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .two-column {
            grid-template-columns: 1fr;
        }
        
        .tech-stack {
            justify-content: center;
        }
        
        .tech-item {
            font-size: 0.8em;
            padding: 0.6rem 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Home page content
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Dashboard Restoran Semarang</h1>
    <p class="main-subtitle">Analisis Cerdas, Prediksi Akurat, Insight Mendalam untuk Bisnis Kuliner</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-number">74</div>
        <div class="stat-label">Jenis Restoran</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">91%</div>
        <div class="stat-label">Akurasi Model</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">2000</div>
        <div class="stat-label">Review Diproses</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">24/7</div>
        <div class="stat-label">Akses</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome-section">
    <h3>Selamat Datang di Dashboard Analytics Restoran Semarang</h3>
    <p>Platform komprehensif untuk analisis data restoran menggunakan teknologi Machine Learning modern. 
    Dapatkan insight tentang karakteristik restoran dan prediksi rating untuk mendukung keputusan bisnis kuliner di Semarang.</p>
    <p><span class="highlight">✨ Fitur Unggulan:</span> Predictive Modeling, Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Exploratory Data Analysis</div>
        <div class="feature-description">
            Jelajahi data restoran dengan visualisasi interaktif. Analisis distribusi rating, 
            jenis restoran, dan fasilitas dengan chart yang responsif dan informatif.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚙️</div>
        <div class="feature-title">Models</div>
        <div class="feature-description">
            Latih dan evaluasi model ML seperti XGBoost. Bandingkan performa model 
            dengan metrik komprehensif dan visualisasi hasil prediksi.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔮</div>
        <div class="feature-title">Rating Prediction</div>
        <div class="feature-description">
            Prediksi rating restoran baru berdasarkan karakteristik seperti jenis restoran, 
            fasilitas, dan jam operasional menggunakan model yang telah dilatih.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="two-column">
    <div class="info-card">
        <h3>📋 Informasi Dataset</h3>
        <p><span class="highlight">Dataset:</span> semarang_resto_dataset.csv</p>
        <p><span class="highlight">Jumlah Records:</span> 5 restoran</p>
        <p><span class="highlight">Features:</span> 10 variabel analisis</p>
        <p><span class="highlight">Coverage:</span> Terbatas pada data tersedia</p>
        <p><span class="highlight">Update:</span> Data statis</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h3>🛠️ Technology Stack</h3>
    <div class="tech-stack">
        <span class="tech-item">🐍 Python</span>
        <span class="tech-item">📊 Streamlit</span>
        <span class="tech-item">🐼 Pandas</span>
        <span class="tech-item">🤖 Scikit-learn</span>
        <span class="tech-item">📈 Plotly</span>
        <span class="tech-item">🔢 NumPy</span>
    </div>
</div>
""", unsafe_allow_html=True)
