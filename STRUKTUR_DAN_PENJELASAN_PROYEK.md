# DOKUMENTASI LENGKAP STRUKTUR INTI DAN PENJELASAN PROYEK
## PILAHCERDAS AI: SISTEM PEMILAHAN SAMPAH OTOMATIS BERBASIS DEEP LEARNING

Dokumen ini disusun untuk memberikan penjelasan komprehensif mengenai latar belakang, arsitektur kecerdasan buatan, alur kerja teknis, dan struktur setiap modul program pada proyek **PilahCerdas AI**.

---

## 1. PENDAHULUAN & TUJUAN PROYEK

Proyek ini dibangun sebagai bagian dari Ujian Tengah Semester (UTS) mata kuliah Kecerdasan Buatan (AI) oleh Kelompok 1.

### Latar Belakang:
- Sampah rumah tangga dan industri kerap tercampur sehingga menghambat proses daur ulang dan menimbulkan bahaya limbah B3.
- Edukasi masyarakat secara konvensional seringkali kurang efektif dalam pemilahan di tempat pembuangan awal.

### Solusi:
Mengembangkan sistem klasifikasi citra cerdas (*Image Classification*) yang dapat membedakan sampah secara instan dan akurat ke dalam 3 kelas:
1. **Organik** (Sampah mudah membusuk: sisa makanan, daun, sayur, buah)
2. **Nonorganik** (Sampah daur ulang: botol plastik, plastik kresek, kaca, kaleng)
3. **Elektronik / E-Waste** (Limbah elektronik: kabel, PCB, baterai, komponen gadget)

---

## 2. STRUKTUR INTI DIREKTORI DAN FILE

```
AI MENDETEKSI GAMBAR/
│
├── dataset/                     # Direktori dataset citra sampah (Elektronik, Nonorganik, Organik)
│
├── app.py                       # Modul Utama Antarmuka Pengguna (Web GUI Streamlit)
├── latih_mesin.py               # Modul Utama Pelatihan Deep Learning (MobileNetV2 Transfer Learning)
├── konversi_jpg.py              # Utilitas Pra-Pemrosesan: Standarisasi format JPEG RGB & filter file korup
├── rename_semua.py              # Utilitas Pra-Pemrosesan: Penyeragaman nama file citra dataset
│
├── model_ai_pintar_final.keras  # Bobot model terlatih format modern Keras (.keras)
├── label_mapping.json           # File JSON pemetaan kelas indeks (0: Elektronik, 1: Nonorganik, 2: Organik)
├── grafik_training.png          # Visualisasi grafik kurva Accuracy & Loss selama pelatihan
│
├── requirements.txt             # Spesifikasi pustaka pihak ketiga (TensorFlow, Streamlit, dll.)
├── .gitignore                   # Daftar filter file yang diabaikan dalam version control Git
├── README.md                    # Dokumentasi utama repositori GitHub
└── STRUKTUR_DAN_PENJELASAN_PROYEK.md  # Dokumen detail analisis proyek ini
```

---

## 3. BEDAH ARSITEKTUR & PENJELASAN KODE PROGRAM

### A. Modul Antarmuka: `app.py`
- **Fungsi**: Menyajikan dashboard interaktif berbasis web untuk pengguna akhir melakukan inferensi (pengujian gambar).
- **Logika & Alur Eksekusi**:
  1. **Konfigurasi Layout & Keamanan Enkoding**: Mengatur `PYTHONUTF8='1'` dan layout `wide` pada Streamlit.
  2. **Model Caching**: Menggunakan `@st.cache_resource` pada fungsi `muat_model()` agar model TensorFlow hanya dimuat satu kali ke memori, mempercepat respon aplikasi.
  3. **Sidebar**: Menampilkan navigasi, informasi ringkas spesifikasi model (MobileNetV2, Akurasi 96.09%, 3 Kategori).
  4. **Komponen Pengujian (Upload & Prediksi)**:
     - Menggunakan `st.file_uploader` untuk menerima file gambar `.jpg`, `.jpeg`, `.png`.
     - Normalisasi orientasi gambar dengan `ImageOps.exif_transpose`.
     - Resize gambar ke resolusi `128x128` piksel dan konversi ke tensor array `(1, 128, 128, 3)`.
     - Melakukan inferensi dengan `model.predict()`.
  5. **Mekanisme Peringatan Keamanan (Safety Filter)**:
     - Jika skor keyakinan tertinggi `< 60%`, sistem menandai `"Bukan Gambar Sampah!"` guna mencegah false positive pada objek asing.
     - Jika skor `> 85%`, memberikan rekomendasi tempat sampah definitif.
  6. **Visualisasi Data & Riwayat**:
     - Kolom kiri: Menampilkan riwayat 5 analisis terakhir dengan timestamp.
     - Kolom kanan: Menampilkan diagram batang distribusi persentase keyakinan untuk ketiga kelas.

---

### B. Modul Pelatihan: `latih_mesin.py`
- **Fungsi**: Melatih arsitektur Convolutional Neural Network (CNN) dari awal hingga menghasilkan model berbobot optimal.
- **Strategi Pemodelan**:
  1. **Akselerasi Perangkat Keras**: Deteksi ketersediaan GPU dan konfigurasi `memory growth` dinamis.
  2. **Data Pipeline tf.data**:
     - Membaca dataset dari folder dengan `image_dataset_from_directory` (pembagian 80% train, 20% validation, `seed=123`).
     - Menggunakan `.prefetch(buffer_size=AUTOTUNE)` untuk pipeline data asinkron berkecepatan tinggi.
  3. **Augmentasi Citra Internal**:
     - `RandomFlip("horizontal")`: Menghadapi variasi orientasi objek.
     - `RandomRotation(0.1)`: Menghadapi variasi sudut pengambilan gambar.
     - `Rescaling(1./127.5, offset=-1)`: Normalisasi nilai piksel ke rentang `[-1, 1]` sesuai standar MobileNetV2.
  4. **Backbone MobileNetV2 Pretrained**:
     - Menggunakan bobot awal dari ImageNet (`weights='imagenet'`).
     - Classifier head: `GlobalAveragePooling2D` -> `Dense(128, relu)` -> `Dropout(0.3)` -> `Dense(3, softmax)`.
  5. **Two-Stage Training**:
     - **Tahap 1 (Warmup / Feature Extraction)**: Backbone dibekukan (`base_model.trainable = False`), melatih classifier head dengan Adam `lr=1e-3` (15 Epoch).
     - **Tahap 2 (Fine-Tuning)**: Membuka 30 layer teratas backbone (`base_model.layers[-30:]`), menjaga layer `BatchNormalization` tetap dibekukan agar mean/variance tidak rusak, menggunakan Adam `lr=1e-5` (10 Epoch).
  6. **Mekanisme Callback**:
     - `EarlyStopping`: Berhenti otomatis jika `val_loss` tidak membaik dalam 3 epoch.
     - `ReduceLROnPlateau`: Membagi dua learning rate jika terjadi stagnansi.
     - `ModelCheckpoint`: Menyimpan model dengan akurasi validasi terbaik (`save_best_only=True`).
  7. **Evaluasi & Visualisasi**:
     - Menguji model terbaik terhadap seluruh data validasi dan menghasilkan `classification_report`.
     - Menggambar kurva akurasi dan loss pada `grafik_training.png`.

---

### C. Modul Sanitasi Data: `konversi_jpg.py`
- **Fungsi**: Mempersiapkan citra dataset agar homogen dan bebas dari anomali format.
- **Mekanisme**:
  - Mengiterasi seluruh subfolder dataset.
  - Membaca citra menggunakan PIL dan mengonversi format warna ke `RGB`. Menghilangkan metadata yang rusak atau format RGBA (4 channel).
  - Menyimpan kembali dengan format ekstensi `.jpg` murni.
  - Menghapus otomatis file yang korup (*corrupted files*).

---

### D. Modul Standarisasi Nama: `rename_semua.py`
- **Fungsi**: Membersihkan penamaan file dalam dataset.
- **Mekanisme**:
  - Mengubah nama file acak dari internet/kamera menjadi penamaan terstruktur berurutan: `img_1.jpg`, `img_2.jpg`, dst.
  - Mencegah error pembacaan path pada sistem operasi akibat spasi, emoji, atau karakter non-ASCII.

---

## 4. HASIL EVALUASI MODEL

- **Akurasi Model**: **96,09%**
- **Kategori Output**:
  1. `Elektronik`
  2. `Nonorganik`
  3. `Organik`
- Model menunjukkan generalisasi yang tinggi berkat kombinasi data augmentasi, dropout 30%, dan fine-tuning bertahap pada arsitektur MobileNetV2.

---

## 5. CARA MENJALANKAN SISTEM SECARA CEPAT

1. **Jalankan Aplikasi Web**:
   ```bash
   streamlit run app.py
   ```
2. **Uji Coba Gambar**:
   - Buka browser di `http://localhost:8501`.
   - Unggah gambar sampah apa pun (misal: botol plastik, daun, atau kabel USB).
   - Klik tombol **Analisis Gambar**.
   - Sistem akan menyajikan hasil klasifikasi, akurasi keyakinan, dan panduan pembuangan sampah.
