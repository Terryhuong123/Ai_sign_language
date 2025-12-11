import cv2
import mediapipe as mp
import numpy as np
import keras
from collections import deque
import time

# ==================== 設定參數 ====================
MODEL_PATH = 'processed_data/model/sign_language_lstm.keras'
LABEL_ENCODER_PATH = 'processed_data/model/label_encoder.npy'
MAX_FRAMES = 30  # 累積幀數
CONFIDENCE_THRESHOLD = 0.3  # 信心度門檻

# ==================== 載入模型 ====================
print("載入模型...")
model = keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
print(f"✓ 模型已載入，共 {len(label_classes)} 個類別")

# ==================== 初始化 MediaPipe ====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================== 輔助函數 ====================
def extract_keypoints_from_frame(results):
    """從單幀中提取關鍵點"""
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            
            if handedness == "Left":
                left_hand = keypoints
            else:
                right_hand = keypoints
    
    combined = left_hand + right_hand
    return combined


def draw_ui(frame, prediction_text, confidence, top_predictions, frame_count, is_recording):
    """繪製使用者介面"""
    h, w = frame.shape[:2]
    
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # 標題
    cv2.putText(frame, "Sign Language Recognition System", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 錄製狀態
    if is_recording:
        status_text = f"Recording: {frame_count}/{MAX_FRAMES}"
        color = (0, 255, 0)  # 綠色
    else:
        status_text = f"Waiting... ({frame_count}/{MAX_FRAMES})"
        color = (100, 100, 100)  # 灰色
    
    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 預測結果
    if prediction_text and confidence >= CONFIDENCE_THRESHOLD:
        # 主要預測
        cv2.putText(frame, f"Prediction: {prediction_text}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Top-3 預測
        y_offset = 200
        cv2.putText(frame, "Top 3 Predictions:", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, (label, conf) in enumerate(top_predictions[:3], 1):
            y_offset += 30
            text = f"{i}. {label} ({conf:.1%})"
            cv2.putText(frame, text, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    elif prediction_text:
        cv2.putText(frame, "Low Confidence - Keep Signing", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # 操作說明
    instructions = [
        "Controls:",
        "SPACE - Start/Stop Recording",
        "R - Reset",
        "Q - Quit"
    ]
    
    y_start = h - 120
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (20, y_start + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def predict_sign(keypoints_sequence):
    """預測手語"""
    # 填充或截斷到 MAX_FRAMES
    keypoints_array = np.array(keypoints_sequence)
    
    if len(keypoints_array) < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - len(keypoints_array), 126))
        keypoints_array = np.vstack([keypoints_array, padding])
    else:
        keypoints_array = keypoints_array[:MAX_FRAMES]
    
    # 預測
    keypoints_array = keypoints_array.reshape(1, MAX_FRAMES, 126)
    predictions = model.predict(keypoints_array, verbose=0)[0]
    
    # 取得 Top-3 預測
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [(label_classes[i], predictions[i]) for i in top_indices]
    
    # 最佳預測
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    predicted_label = label_classes[predicted_class]
    
    return predicted_label, confidence, top_predictions


# ==================== 主程式 ====================
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return
    
    # 設定攝影機解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("🎥 手語辨識系統已啟動")
    print("="*60)
    print("操作說明：")
    print("  [空白鍵] - 開始/停止錄製")
    print("  [R] - 重置")
    print("  [Q] - 退出")
    print("="*60 + "\n")
    
    keypoints_buffer = deque(maxlen=MAX_FRAMES)
    is_recording = False
    prediction_text = ""
    confidence = 0.0
    top_predictions = []
    frame_count = 0
    last_prediction_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取影像")
            break
        
        # 水平翻轉（鏡像）
        frame = cv2.flip(frame, 1)
        
        # 轉換為 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 偵測手部
        results = hands.process(frame_rgb)
        
        # 繪製手部關鍵點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # 錄製模式
            if is_recording:
                keypoints = extract_keypoints_from_frame(results)
                keypoints_buffer.append(keypoints)
                frame_count = len(keypoints_buffer)
                
                # 達到 MAX_FRAMES 時自動預測
                if frame_count >= MAX_FRAMES:
                    current_time = time.time()
                    
                    # 避免過於頻繁預測（每 1 秒最多預測一次）
                    if current_time - last_prediction_time > 1.0:
                        prediction_text, confidence, top_predictions = predict_sign(list(keypoints_buffer))
                        last_prediction_time = current_time
                        
                        print(f"\n預測結果: {prediction_text} ({confidence:.1%})")
                        for i, (label, conf) in enumerate(top_predictions, 1):
                            print(f"  {i}. {label}: {conf:.1%}")
                    
                    # 清空 buffer 準備下次錄製
                    keypoints_buffer.clear()
                    frame_count = 0
                    is_recording = False
        
        # 繪製 UI
        frame = draw_ui(frame, prediction_text, confidence, top_predictions, 
                       frame_count, is_recording)
        
        # 顯示畫面
        cv2.imshow('Sign Language Recognition', frame)
        
        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q 或 ESC
            print("\n退出系統...")
            break
        
        elif key == ord(' '):  # 空白鍵
            if not is_recording:
                print("\n開始錄製手語動作...")
                keypoints_buffer.clear()
                frame_count = 0
                is_recording = True
                prediction_text = ""
            else:
                print("停止錄製")
                is_recording = False
        
        elif key == ord('r') or key == ord('R'):  # R
            print("\n重置")
            keypoints_buffer.clear()
            frame_count = 0
            is_recording = False
            prediction_text = ""
            confidence = 0.0
            top_predictions = []
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✓ 系統已關閉")


if __name__ == "__main__":
    main()