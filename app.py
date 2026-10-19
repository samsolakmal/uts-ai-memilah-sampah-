import os
os.environ['PYTHONUTF8'] = '1'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json
import time
import pandas as pd

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="PilahCerdas AI - Deteksi Sampah", 
    page_icon="♻️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ===== FUNGSI LOAD MODEL =====
@st.cache_resource
def muat_model():
    return tf.keras.models.load_model('model_ai_pintar_final.keras', compile=False)

def muat_kategori():
    with open('label_mapping.json', 'r') as f:
        return json.load(f)

# ===== INISIALISASI SESSION STATE =====
if 'riwayat' not in st.session_state:
    st.session_state.riwayat = []
if 'hasil_terakhir' not in st.session_state:
    st.session_state.hasil_terakhir = None

# ===== LOAD DATA =====
with st.spinner("Memuat model AI..."):
    model = muat_model()
    kategori = muat_kategori()

# ===== SIDEBAR =====
with st.sidebar:
    st.title("♻️ PilahCerdas AI")
    st.markdown("---")
    st.markdown("### 🧭 Menu")
    st.success(" **Beranda**") 
    st.markdown("🧪 Uji Model")
    st.markdown("📁 Manajemen Proyek")
    st.markdown("⚙️ Pengaturan")
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi Model")
    st.info("""
    - **Arsitektur:** MobileNetV2
    - **Akurasi:** 96,09%
    - **Kategori:** 3 (Organik, Nonorganik, Elektronik)
    """)

# ===== HEADER UTAMA =====
st.title("Kelompok 1 - PilahCerdas AI")
st.caption("Sistem Pemilahan Sampah Otomatis Berbasis Deep Learning")
st.markdown("---")

# ===== LAYOUT DASHBOARD (3 KOLOM) =====
kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

# --- KOLOM KIRI: RIWAYAT ---
with kolom_kiri:
    st.subheader("🕒 Riwayat Analisis")
    st.markdown("---")
    
    if len(st.session_state.riwayat) == 0:
        st.caption("Belum ada gambar yang dianalisis.")
    else:
        for item in reversed(st.session_state.riwayat[-5:]):
            with st.container():
                st.markdown(f"**{item['nama']}**")
                st.caption(f"Keyakinan: {item['skor']:.1f}% | {item['waktu']}")
                st.markdown("---")

# --- KOLOM TENGAH: UPLOAD & HASIL ---
with kolom_tengah:
    st.subheader("🧪 Uji Model")
    st.markdown("---")
    
    file_unggah = st.file_uploader("Unggah gambar sampah di sini", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if file_unggah is not None:
        gambar = Image.open(file_unggah)
        gambar = ImageOps.exif_transpose(gambar)
        st.image(gambar, caption="Gambar yang diunggah", use_container_width=True)
        
        st.info("Menunggu analisis... Klik tombol di bawah.")
        
        if st.button(" Analisis Gambar", type="primary", use_container_width=True):
            with st.spinner("AI sedang menganalisis..."):
                gambar_rgb = gambar.convert('RGB')
                gambar_resized = gambar_rgb.resize((128, 128))
                img_array = tf.keras.utils.img_to_array(gambar_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                prediksi = model.predict(img_array, verbose=0)
                indeks_tertinggi = int(np.argmax(prediksi[0]))
                tebakan = kategori[indeks_tertinggi]
                skor_keyakinan = float(np.max(prediksi[0])) * 100
                
                st.session_state.hasil_terakhir = {
                    'nama': tebakan.title(),
                    'skor': skor_keyakinan,
                    'prediksi_full': prediksi[0]
                }
                
                st.session_state.riwayat.append({
                    'nama': tebakan.title(),
                    'skor': skor_keyakinan,
                    'waktu': time.strftime("%H:%M:%S")
                })
                
                st.rerun()
                
    else:
        st.info("Silakan unggah gambar untuk memulai.")
    
    # TAMPILKAN HASIL
    if st.session_state.hasil_terakhir is not None:
        hasil = st.session_state.hasil_terakhir
        tebakan = hasil['nama']
        skor = hasil['skor']
        
        st.markdown("---")
        st.subheader("🤖 Hasil Prediksi AI")
        
        if skor < 60.0:
            st.error(f"❌ **Bukan Gambar Sampah!**\nKeyakinan: {skor:.1f}% (Di bawah batas aman)")
        else:
            st.markdown(f"### {tebakan}")
            st.metric(label="Tingkat Keyakinan", value=f"{skor:.2f}%")
            
            if skor > 85.0:
                st.success(f"Buang ke tong **{tebakan.upper()}**")
            elif skor > 70.0:
                st.info(f"Kemungkinan besar ini adalah sampah **{tebakan.title()}**.")
            else:
                st.warning(f"AI mendeteksi **{tebakan.title()}** tapi dengan keyakinan sedang.")

# --- KOLOM KANAN: STATISTIK ---
with kolom_kanan:
    st.subheader("🎯 Akurasi Keseluruhan")
    st.markdown("---")
    st.metric(label="Akurasi Model", value="96,09%", delta="Keyakinan Tinggi")
    st.markdown("---")
    
    st.subheader("📊 Distribusi Keyakinan")
    st.markdown("---")
    
    if st.session_state.hasil_terakhir is not None:
        prediksi_full = st.session_state.hasil_terakhir['prediksi_full']
        
        data_chart = {
            'Kategori': [k.title() for k in kategori],
            'Keyakinan (%)': [float(p) * 100 for p in prediksi_full]
        }
        df_chart = pd.DataFrame(data_chart).sort_values('Keyakinan (%)', ascending=True)
        
        st.bar_chart(df_chart.set_index('Kategori'))
    else:
        st.caption("Unggah gambar untuk melihat distribusi keyakinan.")

# ===== FOOTER =====
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8rem;'>© 2024 Proyek PilahCerdas AI | Kelompok 1</div>", unsafe_allow_html=True)