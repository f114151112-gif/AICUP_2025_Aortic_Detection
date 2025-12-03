import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

def apply_laplacian(image_path, output_path, mode='sharpen'):
    # 讀取圖片 (灰階)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 高斯模糊降噪 (減少雜訊對 Laplacian 的影響)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Laplacian 運算子 (偵測邊緣)
    # 使用 CV_16S 防止數值溢出，再轉回 uint8
    laplacian = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(laplacian)

    if mode == 'blend':
        # 銳化效果：原圖 + 邊緣
        # 這裡的權重可以調整：
        # alpha=1.0 (原圖保留程度)
        # beta=0.5 (邊緣疊加程度，越高越銳利)
        sharpened = cv2.addWeighted(img, 1.0, edges, 0.5, 0)
        cv2.imwrite(str(output_path), sharpened)
    else:
        # 預設：只存 Laplacian 邊緣圖
        cv2.imwrite(str(output_path), edges)

def main():
    parser = argparse.ArgumentParser(description="Apply Laplacian Sharpening to a dataset.")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--mode", type=str, default="blend", choices=["edge_only", "blend"], help="Output mode: 'edge_only' or 'blend' (Sharpening)")
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)
    mode = args.mode

    if output_root.exists():
        print(f"Cleaning existing output directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    # 遞迴找出所有檔案
    all_files = list(source_root.rglob("*"))
    
    print(f"Found {len(all_files)} files in {source_root}. Processing...")

    for file_path in tqdm(all_files):
        if not file_path.is_file():
            continue
            
        # 保持原本的資料夾結構
        rel_path = file_path.relative_to(source_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # 如果是圖片，做 Laplacian
            apply_laplacian(file_path, out_path, mode)
        elif file_path.suffix.lower() == '.txt':
            # 如果是標籤檔，直接複製
            shutil.copy2(file_path, out_path)
        else:
            # 其他檔案 (如 yaml) 複製
            shutil.copy2(file_path, out_path)

    print("Done! Laplacian sharpening applied and labels copied.")
    print(f"Output saved to: {output_root}")

if __name__ == "__main__":
    main()
