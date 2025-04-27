# Yolo Mirror 虛擬試衣專案

## 專案簡介
本專案結合 YOLO 姿態辨識與虛擬服裝貼合技術，實現即時虛擬試衣功能。支援即時串流、服裝動態更換，適用於互動展示、線上試衣等應用場景。

## 特色與功能
- YOLO 姿態偵測，精準取得人體關鍵點
- 支援上衣、下身服裝自動貼合
- MJPEG 串流即時顯示
- 多執行緒高效處理
- 動態更換服裝圖檔
- 支援影片或攝影機輸入

## 安裝方式

1. 下載專案
   ```bash
   git clone <repo-url>
   cd yolo_mirror
   ```

2. 安裝 Python 依賴
   ```bash
   pip install -r requirements.txt
   ```

3. 準備模型與服裝圖檔  
   - 將 YOLO 模型（如 `yolo11m-pose.pt`）放入專案資料夾  
   - 準備服裝圖檔（PNG 格式，建議透明背景）

## 使用說明

執行主程式：
```bash
python yolo_pose_app.py
```

可自訂服裝圖檔：
```python
app = create_yolo_app(
    clothes_path_top="tshirt1.png",
    clothes_path_bottom="pants1.png"
)
app.run()
```

瀏覽器開啟 `http://<server-ip>:5050` 觀看即時串流。

## 目錄結構說明

```
yolo_mirror/
    yolo_pose_app.py      # 主程式
    *.pt                  # YOLO 模型檔
    *.png                 # 服裝圖檔
    original.jpg          # 測試圖片
    ...
```

## 依賴與需求

- Python 3.8+
- OpenCV
- Pillow
- ultralytics
- mjpeg_streamer

## 常見問題

- Q: 執行時找不到攝影機？  
  A: 請確認 camera_index 設定正確，或改用影片檔測試。

- Q: 服裝貼合位置不正確？  
  A: 請確認服裝圖檔比例與人體關鍵點對應。

