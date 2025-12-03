# predict_aortic_yolov12.py (Version 3: Blend)
from ultralytics import YOLO
from pathlib import Path

# 1. 權重位置
# 1. 權重位置 (改回您分數最高的那個模型)
BEST_WEIGHTS = r"C:\ai_cup_dataset\runs_aicup\yolov12n_laplacian\weights\best.pt"

# 2. 測試圖片根目錄 (融合後的圖片)
TEST_ROOT = Path(r"C:\ai_cup_dataset\test_laplacian")

# 3. 輸出位置
OUTPUT_DIR = Path(r"C:\ai_cup_dataset\predict_txt")
DEVICE = 0
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".pfm"}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = OUTPUT_DIR / "merged_laplacian.txt" # 測試 0.01

    print("載入模型權重：", BEST_WEIGHTS)
    model = YOLO(BEST_WEIGHTS)

    print("掃描測試圖片中...", TEST_ROOT)
    image_paths = sorted(p for p in TEST_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        print("⚠ 找不到圖片")
        return

    total = len(image_paths)
    print(f"找到 {total} 張圖片，開始預測...")

    source_pattern = str(TEST_ROOT / "**" / "*.*")
    results_gen = model.predict(
        source=source_pattern,
        imgsz=640,
        batch=32,
        device=DEVICE,
        stream=True,
        verbose=False,
        save=False,
        augment=False, # <--- 修正：關閉 TTA！因為心臟有左右之分，翻轉會導致錯誤
        conf=0.01,     # 使用者要求測試 0.01
        iou=0.5,       # 標準 NMS
        max_det=3      # 強制只留前 3 名
    )

    lines = []
    processed = 0
    for r in results_gen:
        processed += 1
        img_path = Path(r.path)
        image_id = img_path.stem
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                line = f"{image_id} 0 {conf:.4f} {round(x_min)} {round(y_min)} {round(x_max)} {round(y_max)}"
                lines.append(line)
        
        if processed % 50 == 0:
            print(f"[progress] {processed}/{total}", end='\r')

    lines.sort()
    merged_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== 完成！已寫出 {len(lines)} 行到 {merged_path} ===")

if __name__ == "__main__":
    main()
