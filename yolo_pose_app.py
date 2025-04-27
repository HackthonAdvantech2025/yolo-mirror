import cv2
from mjpeg_streamer import MjpegServer, Stream
from ultralytics import YOLO
import time
import numpy as np
from PIL import Image
import functools
from datetime import datetime
import threading
import queue

def time_it(func):
    """
    函數裝飾器：記錄函數執行時間。
    Args:
        func: 被裝飾的函數
    Returns:
        包裝後的函數
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Completed {func.__name__}, execution time: {execution_time:.2f} milliseconds")
        return result
    return wrapper

class YoloPoseApp:
    def __init__(self, model_path, clothes_path_top, clothes_path_bottom=None, camera_index=10, stream_size=(1080, 1920), stream_quality=60, stream_fps=30):
        self.model = YOLO(model_path)
        self.stream = Stream("mjpeg", size=stream_size, quality=stream_quality, fps=stream_fps)
        self.server = MjpegServer("0.0.0.0", 5050)
        self.server.add_stream(self.stream)
        self.server.start()
        self.cap = cv2.VideoCapture(camera_index)
        self.clothes_img_top = Image.open(clothes_path_top).convert("RGBA")
        if clothes_path_bottom:
            self.clothes_img_bottom = Image.open(clothes_path_bottom).convert("RGBA")
        else:
            self.clothes_img_bottom = None
        self.prev_time = time.time()
        self.frame_time_array = [self.prev_time]
        self.max_fps_samples = 10
        self.preprocess_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.raw_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.read_thread = threading.Thread(target=self._read_worker, daemon=True)
        self.preprocess_thread = threading.Thread(target=self._preprocess_worker, daemon=True)
        self.infer_thread.start()
        print(f"[YoloPoseApp] 模型載入中，等待 10s中...")
        time.sleep(10)  # 等待模型初始化
        print(f"[YoloPoseApp] 開始讀取攝影機...")
        self.display_thread.start()
        self.read_thread.start()
        self.preprocess_thread.start()

    def set_clothes_path(self, clothes_path, part="top"):
        """
        動態更換衣服圖檔
        part: "top" 或 "bottom"
        """
        img = Image.open(clothes_path).convert("RGBA")
        if part == "top":
            self.clothes_img_top = img
            print(f"[YoloPoseApp] 已更換上衣圖：{clothes_path}")
        elif part == "bottom":
            self.clothes_img_bottom = img
            print(f"[YoloPoseApp] 已更換下身圖：{clothes_path}")
        else:
            print(f"[YoloPoseApp] part 參數錯誤，必須為 'top' 或 'bottom'")

    # -------------------- 衣服貼合相關 --------------------
    def _apply_clothes(self, result, annotated_pil):
        """
        根據肩膀、手肘、臀部等關鍵點自動貼合上身服裝（如T恤、外套）
        流程：
        1. 檢查必須的關鍵點是否存在
        2. 計算上邊界（肩膀中點）
        3. 計算下邊界（臀部中點）
        4. 計算左右邊界（肩膀與手肘的中間點）
        5. 確保邊界範圍並計算寬高
        6. 調整大小並貼上衣服
        """
        keypoints_data = result.keypoints
        if keypoints_data is not None and len(keypoints_data) > 0 and self.clothes_img_top is not None:
            kp = keypoints_data.data[0].cpu().numpy()
            if not self._is_upper_keypoints_valid(kp):
                print("上衣關鍵點不足，無法貼合上衣服裝")
                return annotated_pil
            top_y = self._calc_upper_top_y(kp)
            bottom_y = self._calc_upper_bottom_y(kp)
            left_x, right_x = self._calc_upper_left_right_x(kp)
            left_x = max(0, left_x)
            right_x = min(1080, right_x)
            clothes_width = max(1, right_x - left_x)
            clothes_height = max(1, bottom_y - top_y)
            clothes_resized = self.clothes_img_top.resize((clothes_width, clothes_height))
            offset_x = left_x
            offset_y = top_y
            print(f"上衣貼上成功, 位置: ({offset_x}, {offset_y}), 大小: ({clothes_width}, {clothes_height})")
            annotated_pil.paste(clothes_resized, (offset_x, offset_y), clothes_resized)
        return annotated_pil

    def _apply_bottom_clothes(self, result, annotated_pil):
        """
        根據髖部與膝蓋關鍵點自動貼合下身服裝（如褲子、裙子）
        流程：
        1. 檢查必須的關鍵點是否存在
        2. 計算上邊界（髖部中點）
        3. 計算下邊界（膝蓋中點）
        4. 計算左右邊界（四點的最左與最右）
        5. 確保邊界範圍並計算寬高
        6. 調整大小貼上下身服裝
        """
        keypoints_data = result.keypoints
        if keypoints_data is not None and len(keypoints_data) > 0 and self.clothes_img_bottom is not None:
            kp = keypoints_data.data[0].cpu().numpy()
            if not self._is_lower_keypoints_valid(kp):
                print("下身關鍵點不足，無法貼合下身服裝")
                return annotated_pil
            top_y = self._calc_lower_top_y(kp)
            bottom_y = self._calc_lower_bottom_y(kp)
            left_x, right_x = self._calc_lower_left_right_x(kp)
            left_x = max(0, left_x)
            right_x = min(1080, right_x)
            clothes_width = max(1, right_x - left_x)
            clothes_height = max(1, bottom_y - top_y)
            clothes_resized = self.clothes_img_bottom.resize((clothes_width, clothes_height))
            offset_x = left_x
            offset_y = top_y
            print(f"下身貼上成功, 位置: ({offset_x}, {offset_y}), 大小: ({clothes_width}, {clothes_height})")
            annotated_pil.paste(clothes_resized, (offset_x, offset_y), clothes_resized)
        return annotated_pil

    # -------------------- 關鍵點檢查與邊界輔助方法（統一管理） --------------------
    def _is_upper_keypoints_valid(self, kp):
        # 上身必須有左右肩膀、左右臀部
        return kp.shape[0] > 12 and all(np.any(kp[i] != 0) for i in [5, 6, 11, 12])

    def _is_lower_keypoints_valid(self, kp):
        # 下身必須有左右髖部與左右膝蓋
        return kp.shape[0] > 14 and all(np.any(kp[i] != 0) for i in [11, 12, 13, 14])

    def _calc_upper_top_y(self, kp):
        # 上邊界：左右肩膀中點
        return int((kp[5][1] + kp[6][1]) / 2)

    def _calc_upper_bottom_y(self, kp):
        # 下邊界：左右臀部中點
        return int((kp[11][1] + kp[12][1]) / 2)

    def _calc_upper_left_right_x(self, kp):
        # 左右邊界：肩膀與手肘的中間點
        left_x = int((kp[5][0] + kp[7][0]) / 2) if np.any(kp[7] != 0) else int(kp[5][0])
        right_x = int((kp[6][0] + kp[8][0]) / 2) if np.any(kp[8] != 0) else int(kp[6][0])
        return left_x, right_x

    def _calc_lower_top_y(self, kp):
        # 上邊界：左右髖部中點
        return int((kp[11][1] + kp[12][1]) / 2)

    def _calc_lower_bottom_y(self, kp):
        # 下邊界：左右膝蓋中點
        return int((kp[13][1] + kp[14][1]) / 2)

    def _calc_lower_left_right_x(self, kp):
        # 左右邊界：髖部與膝蓋的最左與最右
        xs = [kp[11][0], kp[12][0], kp[13][0], kp[14][0]]
        return int(min(xs)), int(max(xs))

    # -------------------- FPS 與顯示相關 --------------------
    def _update_fps(self):
        curr_time = time.time()
        self.frame_time_array.append(curr_time)
        if len(self.frame_time_array) > self.max_fps_samples:
            self.frame_time_array.pop(0)
        instant_fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        avg_fps = 1 / np.mean(np.diff(self.frame_time_array))
        print(f"Instant FPS: {instant_fps:.2f}, Avg FPS: {avg_fps:.2f}")
        return instant_fps, avg_fps

    def _draw_fps_and_show(self, frame, instant_fps, avg_fps):
        cv2.putText(frame, f"FPS: {instant_fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # 可加上 cv2.imshow 或其他顯示功能（如有需要）

    def save_and_stream_frame(self, frame, path="output.jpg"):
        # cv2.imwrite(path, frame)
        self.stream.set_frame(frame)

    # -------------------- 執行緒與主流程 --------------------
    def _read_worker(self):
        while not self.stop_event.is_set() and self.cap.isOpened():
            success, frame = self.read_frame()
            if not success:
                break
            try:
                self.raw_queue.put(frame, timeout=0.1)
            except queue.Full:
                continue

    def _preprocess_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.raw_queue.get(timeout=0.1)
                pre_frame = self.preprocess_frame(frame)
                self.preprocess_queue.put(pre_frame, timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in preprocess: {e}")
                continue

    def _inference_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.preprocess_queue.get(timeout=0.1)
                results = self.model(frame)
                self.result_queue.put((frame, results))
            except queue.Empty:
                continue
    
    def _display_worker(self):
        while not self.stop_event.is_set():
            try:
                frame, results = self.result_queue.get(timeout=0.1)
                self._postprocess_and_show(frame, results)
            except queue.Empty:
                continue

    @time_it
    def _postprocess_and_show(self, frame, results):
        # 合併 plot_result 與 convert_to_pil
        annotated_frame = results[0].plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        for result in results:
            self._print_summary(result)
            annotated_pil = self._apply_clothes(result, annotated_pil)
            annotated_pil = self._apply_bottom_clothes(result, annotated_pil)
        annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGBA2BGR)
        instant_fps, avg_fps = self._update_fps()
        self._draw_fps_and_show(annotated_frame, instant_fps, avg_fps)
        self.save_and_stream_frame(annotated_frame)

    @time_it
    def read_frame(self):
        success, frame = self.cap.read()
        return success, frame

    @time_it
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (1080, 1920))
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @time_it
    def run(self):
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)  # 主執行緒保持簡潔，僅負責監控stop_event
        finally:
            self.stop_event.set()
            self.read_thread.join()
            self.preprocess_thread.join()
            self.infer_thread.join()
            self.display_thread.join()
            self.cap.release()
            cv2.destroyAllWindows()

    def _print_summary(self, result):
        summary = result.summary()
        for obj in summary:
            print(f"Object: {obj['name']}, confidence: {obj['confidence']:.2f}")

def create_yolo_app(clothes_path_top=None, clothes_path_bottom=None):
    """
    建立新的 YoloPoseApp 實例，可指定衣服圖路徑
    """
    import os
    if clothes_path_top is None:
        clothes_path_top = os.path.expanduser("/home/icam-540/yolo_mirror/tshirt1.png")
    return YoloPoseApp(
        model_path=os.path.expanduser("/home/icam-540/yolo_mirror/yolo11m-pose.pt"),
        clothes_path_top=clothes_path_top,
        clothes_path_bottom=clothes_path_bottom,
        camera_index='/home/icam-540/fast_mirror/pic/people_demo.mp4',
        stream_size=(1080, 1920),
        stream_quality=75,
        stream_fps=30
    )

if __name__ == "__main__":
    # 建立主應用物件，指定模型、衣服圖、攝影機等參數
    app = create_yolo_app(
        clothes_path_top="/home/icam-540/yolo_mirror/tshirt1.png",
        clothes_path_bottom="/home/icam-540/yolo_mirror/pants1.png"
    )
    app.run()