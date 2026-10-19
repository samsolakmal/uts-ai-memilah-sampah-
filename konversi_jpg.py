import os
from PIL import Image

# Path ke dataset asli (sesuai setup junction link kamu)
dataset_path = r'D:\dataset'

print("🔄 Memaksa konversi semua gambar sampah ke format JPEG standar...")
converted_count = 0
deleted_count = 0

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        print(f"📂 Memproses folder: {folder_name}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue
            
            try:
                # Buka gambar
                with Image.open(file_path) as img:
                    # Konversi ke RGB (buang alpha channel biar bisa jadi JPEG)
                    img = img.convert('RGB')
                    
                    # Ganti ekstensi jadi .jpg
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_path = os.path.join(folder_path, new_filename)
                    
                    # Simpan sebagai JPEG standar
                    img.save(new_path, 'JPEG')
                    
                    # Hapus file lama kalau namanya beda
                    if new_path != file_path:
                        os.remove(file_path)
                        
                    converted_count += 1
            except Exception as e:
                # Kalau gagal dibuka (file benar-benar rusak), hapus
                print(f"🗑️ File rusak dihapus: {filename}")
                try:
                    os.remove(file_path)
                except:
                    pass
                deleted_count += 1

print(f"\n✅ SELESAI!")
print(f"🔄 {converted_count} gambar berhasil dikonversi ke JPEG.")
print(f"🗑️ {deleted_count} file rusak dihapus.")
print("\n Sekarang jalankan: python latih_mesin.py")