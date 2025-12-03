# -*- coding: utf-8 -*-
import torch
from ultralytics import YOLO


DATA_YAML = r"C:/ai_cup_dataset/aortic_laplacian.yaml"

def main():
    print("=== PyTorch / CUDA 狀態 ===")
    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        device = "cpu"

    model = YOLO("yolov12n.pt")
    
    results = model.train(
        data=DATA_YAML,
        epochs=200,    # 還原冠軍設定
        imgsz=640,
        batch=-1,
        device=device,
        workers=2,
        cache=True,
        project="runs_aicup",
        name="yolov12n_laplacian", # 區分名稱
        patience=50,   # 還原冠軍設定
        cos_lr=True,
        close_mosaic=10,
        fliplr=0.0,
        mixup=0.1,
        # 新增預測參數 (讓驗證時的指標更接近繳交結果)
        conf=0.01,
        iou=0.5,
        max_det=3,
    )

    print("=== 訓練結束 ===")
    print("log / 權重存放位置：", results.save_dir)

if __name__ == "__main__":
    main()
