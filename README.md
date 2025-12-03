
# 日期: 2025/12/03

## 1. 核心結論

- **模型檔案**: `c:\ai_cup_dataset\runs_aicup\yolov12n_laplacian\weights\best.pt`
- **建立時間**: 2025/11/24 04:32:28 AM 
- **參數設定**:
  - `conf`: **0.01** 
  - `iou`: 0.5
  - `max_det`: 3

## 2. 驗證證據
我們比對了重新執行的結果與原始繳交檔 (`merged_laplacian.txt`)：

| 項目 | 原始繳交檔 | 重現檔 (Conf=0.01)  |
| :--- | :--- | :--- | :--- |
| **行數** | **4037 行** | **4033 行** (僅差 4 行，吻合度 99.9%) | 
| **內容** | patient0051... 0.03621 | patient0051... 0.03621 | (數值一致) |

## 3. 檔案說明 
在此資料夾中備份了以下檔案：
1. **`predict_aortic_yolov12.py`**: 預測腳本 (已設定正確參數)。
2. **`train_aortic_yolov12.py`**: 訓練腳本 ( Epochs=200, Patience=50)。
3. **`preprocess_laplacian.py`**: 前處理腳本 (Laplacian Blend)。
4. **`split_train_val.py`**: 資料切分腳本 (驗證比例約 8%)。
5. **`aortic_laplacian.yaml`**: 資料集設定檔。
6. **`merged_laplacian_conf001_iou05_reproduce.txt`**: 剛剛驗證成功的預測結果。

## 4. 完整重現流程 (Step-by-Step)

### 步驟一：資料前處理 (產生銳化資料集)
執行以下指令產生 `train_laplacian`, `val_laplacian`, `test_laplacian` 資料夾：
```bash
python preprocess_laplacian.py --source train --output train_laplacian --mode blend
python preprocess_laplacian.py --source val --output val_laplacian --mode blend
python preprocess_laplacian.py --source test --output test_laplacian --mode blend
```

### 步驟二：資料切分 (若需要)
*注意：若您已經有 `val` 資料夾，通常不需要重新切分，除非您想改變驗證集。*
```bash
python split_train_val.py
```
(本專案使用 `VAL_RATIO = 0.08`，即約 8% 資料作為驗證集)

### 步驟三：訓練模型
```bash
python train_aortic_yolov12.py
```
*   會讀取 `aortic_laplacian.yaml`
*   訓練參數：Epochs=200, Patience=50
*   模型輸出位置：`runs_aicup/yolov12n_laplacian`

### 步驟四：預測結果
```bash
python predict_aortic_yolov12.py
```
*   會讀取 `runs_aicup/yolov12n_laplacian/weights/best.pt`
*   預測參數：`conf=0.01`, `iou=0.5`, `max_det=3`
*   輸出檔案：`merged_laplacian_final.txt`

## 5. 前處理參數細節
根據程式碼預設值與資料夾名稱 (`train_laplacian`)，當時的前處理設定應為：
- **Mode**: `blend` (混合模式)
- **Alpha (原圖權重)**: `1.0`
- **Beta (邊緣權重)**: `0.5`
- **Kernel Size**: `3`
- **Gaussian Blur**: `(3, 3)`




