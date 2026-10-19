import os
os.environ['PYTHONUTF8'] = '1'  # PERISAI UTF-8 - WAJIB DI PALING ATAS!

import tensorflow as tf
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ===== KONFIGURASI GPU =====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🚀 Ditemukan {len(gpus)} GPU. Mengaktifkan memory growth...")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ GPU tidak terdeteksi. Training akan menggunakan CPU.")
    print("💡 CPU Intel i5-12450HX kamu cukup kencang untuk MobileNetV2!\n")

# ===== KONFIGURASI UTAMA =====
folder_dataset = 'dataset'  # Mengacu ke junction link -> D:\dataset
ukuran_gambar = (128, 128)
batch_size = 32  # Aman untuk RAM 12GB

print("=" * 60)
print("🧠 MEMULAI PELATIHAN MODEL DEWA (MobileNetV2 + Fine-Tuning)")
print("♻️  Dataset: Deteksi Sampah (Elektronik, Nonorganik, Organik)")
print("=" * 60)

# ===== 1. LOAD DATA =====
print("\n📂 Memuat dataset...")
try:
    dataset_latih = tf.keras.utils.image_dataset_from_directory(
        folder_dataset, image_size=ukuran_gambar, batch_size=batch_size,
        label_mode='int', validation_split=0.2, subset="training", seed=123)
    dataset_val = tf.keras.utils.image_dataset_from_directory(
        folder_dataset, image_size=ukuran_gambar, batch_size=batch_size,
        label_mode='int', validation_split=0.2, subset="validation", seed=123)
    print("✅ Dataset berhasil dimuat!")
except Exception as e:
    print(f"❌ Error memuat data: {e}")
    print("💡 Coba jalankan 'python rename_semua.py' dulu untuk merename file.")
    sys.exit(1)

# ===== 2. SIMPAN LABEL MAPPING =====
nama_kategori = dataset_latih.class_names
print(f"📊 Total kategori: {len(nama_kategori)} jenis")
print("📋 Daftar kategori:")
for i, nama in enumerate(nama_kategori):
    print(f"   [{i}] {nama}")

with open('label_mapping.json', 'w') as f:
    json.dump(nama_kategori, f, indent=2)
print("✅ Label mapping disimpan ke 'label_mapping.json'\n")

# ===== 3. OPTIMASI PIPELINE =====
AUTOTUNE = tf.data.AUTOTUNE
dataset_latih = dataset_latih.prefetch(buffer_size=AUTOTUNE)
dataset_val = dataset_val.prefetch(buffer_size=AUTOTUNE)

# ===== 4. BANGUN MODEL (FUNCTIONAL API) =====
print("🏗️  Membangun arsitektur model (Functional API)...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Dibekukan untuk Tahap 1

inputs = tf.keras.Input(shape=(128, 128, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)  # Training=False untuk Tahap 1 (aman untuk BatchNorm)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(nama_kategori), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===== 5. CALLBACKS =====
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'model_ai_pintar_final.keras', monitor='val_accuracy', 
        save_best_only=True, verbose=1
    )
]

# ===== 6. TAHAP 1: TRAINING HEAD =====
print("\n" + "=" * 60)
print("🔥 TAHAP 1: Melatih classifier head (base model dibekukan)")
print("=" * 60)
history1 = model.fit(
    dataset_latih, validation_data=dataset_val,
    epochs=15, callbacks=callbacks
)

# ===== 7. TAHAP 2: FINE-TUNING =====
print("\n" + "=" * 60)
print("🎯 TAHAP 2: Fine-tuning 30 layer teratas MobileNetV2")
print("=" * 60)

base_model.trainable = True
# Bekukan semua layer kecuali 30 layer terakhir
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Bekukan BatchNormalization secara eksplisit di 30 layer terakhir
for layer in base_model.layers[-30:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

history2 = model.fit(
    dataset_latih, validation_data=dataset_val,
    epochs=10, callbacks=callbacks
)

# ===== 8. EVALUASI DETAIL =====
print("\n" + "=" * 60)
print("📋 EVALUASI DETAIL PER KELAS (Menggunakan Model Terbaik)")
print("=" * 60)

# Load model terbaik yang sudah disimpan oleh ModelCheckpoint
model_eval = tf.keras.models.load_model('model_ai_pintar_final.keras', compile=False)

y_true, y_pred = [], []
for x, y in dataset_val:
    pred = model_eval.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(pred, axis=1))

print(classification_report(y_true, y_pred, target_names=nama_kategori, zero_division=0))

# ===== 9. PLOTTING GRAFIK =====
print("📊 Membuat grafik training...")
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training dan Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training dan Validation Loss')
plt.savefig('grafik_training.png')
plt.close()
print("✅ Grafik disimpan sebagai 'grafik_training.png'\n")

# ===== 10. SELESAI =====
print("=" * 60)
print("🎉 PELATIHAN SELESAI!")
print("✅ Model terbaik tersimpan di: model_ai_pintar_final.keras")
print("✅ Label mapping tersimpan di: label_mapping.json")
print("✅ Grafik tersimpan di: grafik_training.png")
print("=" * 60)