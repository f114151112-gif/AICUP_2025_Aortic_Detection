import os
import random
import shutil
import sys

# ===== 這裡改成你的 datasets 根目錄 =====
# ===== 這裡改成你的 datasets 根目錄 =====
BASE = os.getcwd()

# train 的 images / labels 位置
TRAIN_IMG_ROOT = os.path.join(BASE, "train", "images")
TRAIN_LBL_ROOT = os.path.join(BASE, "train", "labels")

# val 的 images / labels 位置（如果不存在會自動建立）
VAL_IMG_ROOT = os.path.join(BASE, "val", "images")
VAL_LBL_ROOT = os.path.join(BASE, "val", "labels")

os.makedirs(VAL_IMG_ROOT, exist_ok=True)
os.makedirs(VAL_LBL_ROOT, exist_ok=True)

# 🔒 重置檢查：如果 val/images 裡已經有東西，先把資料搬回 train
if os.listdir(VAL_IMG_ROOT):
    print("⚠ 偵測到 val/images 已經有資料，正在執行重置 (Reset)...")
    print("   將所有 val 資料搬回 train，以便重新切分。")
    
    # 取得 val 底下的 patient
    val_patients_existing = [d for d in os.listdir(VAL_IMG_ROOT) if os.path.isdir(os.path.join(VAL_IMG_ROOT, d))]
    
    for patient in val_patients_existing:
        src_img_dir = os.path.join(VAL_IMG_ROOT, patient)
        src_lbl_dir = os.path.join(VAL_LBL_ROOT, patient)
        
        dst_img_dir = os.path.join(TRAIN_IMG_ROOT, patient)
        dst_lbl_dir = os.path.join(TRAIN_LBL_ROOT, patient)
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        # 搬圖片
        for fname in os.listdir(src_img_dir):
            shutil.move(os.path.join(src_img_dir, fname), os.path.join(dst_img_dir, fname))
            
        # 搬標籤
        if os.path.exists(src_lbl_dir):
            for fname in os.listdir(src_lbl_dir):
                shutil.move(os.path.join(src_lbl_dir, fname), os.path.join(dst_lbl_dir, fname))
            os.rmdir(src_lbl_dir)
            
        # 刪除空的 patient 資料夾
        os.rmdir(src_img_dir)
        
    print("✅ 重置完成！所有資料已回到 train。")

# 取得所有 patient 資料夾名稱（例如 patient0001, patient0002,...）
patients = [
    d for d in os.listdir(TRAIN_IMG_ROOT)
    if os.path.isdir(os.path.join(TRAIN_IMG_ROOT, d))
]

print("在 train/images 底下找到 patient 數量：", len(patients))

# 想切多少當 val？這裡先用 20%
VAL_RATIO = 0.08
random.seed(42)  # 固定亂數種子，重跑結果一樣（第一次）

val_count = max(1, int(len(patients) * VAL_RATIO))
val_patients = set(random.sample(patients, val_count))

print("將以下 patient 當作 val：")
for p in sorted(val_patients):
    print("  ", p)

# 開始搬動這些 patient 的圖片 & 標註
for patient in patients:
    src_img_dir = os.path.join(TRAIN_IMG_ROOT, patient)
    src_lbl_dir = os.path.join(TRAIN_LBL_ROOT, patient)

    # 判斷這個病人要不要進 val
    if patient in val_patients:
        dst_img_dir = os.path.join(VAL_IMG_ROOT, patient)
        dst_lbl_dir = os.path.join(VAL_LBL_ROOT, patient)
    else:
        # 留在 train，就不用動
        continue

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # 搬圖片
    for fname in os.listdir(src_img_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        src_img_path = os.path.join(src_img_dir, fname)
        dst_img_path = os.path.join(dst_img_dir, fname)
        shutil.move(src_img_path, dst_img_path)

        # 對應的 label 檔名：同名 + .txt
        stem, _ = os.path.splitext(fname)
        src_txt_path = os.path.join(src_lbl_dir, stem + ".txt")
        if os.path.exists(src_txt_path):
            dst_txt_path = os.path.join(dst_lbl_dir, stem + ".txt")
            shutil.move(src_txt_path, dst_txt_path)

    print(f"已搬到 val：{patient}")

print("✅ 切分完成！記得檢查一下 val/images 和 val/labels 裡的東西。")
