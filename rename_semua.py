import os

dataset_path = r'D:\dataset'

print(" MEMULAI JURUS NUKE: Merename semua file dataset sampah...")
total_rename = 0

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    if os.path.isdir(folder_path):
        count = 1
        for filename in os.listdir(folder_path):
            old_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(old_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    ext = '.jpg' 
                
                new_name = f"img_{count}{ext}"
                new_path = os.path.join(folder_path, new_name)
                
                try:
                    os.rename(old_path, new_path)
                    count += 1
                    total_rename += 1
                except Exception as e:
                    print(f"⚠️ Gagal rename {filename}: {e}")

print(f"\n✅ SELESAI! {total_rename} file telah diamankan.")
print("🚀 Sekarang jalankan: python latih_mesin.py")