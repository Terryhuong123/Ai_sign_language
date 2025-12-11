import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys

# 強制即時輸出
sys.stdout.flush()

# 初始化 MediaPipe
mp_hands = mp.solutions.hands

def extract_hand_keypoints(video_path, max_frames=30):
    """從影片中提取手部關鍵點"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"無法開啟影片: {video_path}")
    
    keypoints_sequence = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
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
            keypoints_sequence.append(combined)
            frame_count += 1
        
        cap.release()
    
    keypoints_array = np.array(keypoints_sequence)
    
    if len(keypoints_array) < max_frames:
        padding = np.zeros((max_frames - len(keypoints_array), 126))
        keypoints_array = np.vstack([keypoints_array, padding])
    else:
        keypoints_array = keypoints_array[:max_frames]
    
    return keypoints_array


def process_dataset(excel_file, videos_folder, output_folder, max_frames=30):
    """處理整個資料集"""
    
    print("=" * 80)
    print("🚀 手語影片關鍵點提取程式")
    print("=" * 80)
    sys.stdout.flush()
    
    # 讀取 Excel
    if not os.path.exists(excel_file):
        print(f"❌ 找不到 Excel 檔案: {excel_file}")
        return None, None
    
    df = pd.read_excel(excel_file)
    print(f"\n✓ 成功讀取 Excel")
    print(f"  總資料筆數: {len(df)}")
    sys.stdout.flush()
    
    # 檢查影片資料夾
    if not os.path.exists(videos_folder):
        print(f"❌ 找不到影片資料夾: {videos_folder}")
        return None, None
    
    video_files = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')]
    print(f"  影片資料夾: {len(video_files)} 個 .mp4 檔案")
    sys.stdout.flush()
    
    # 建立輸出資料夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 開始處理
    print(f"\n" + "=" * 80)
    print(f"開始處理 {len(df)} 個影片...")
    print("=" * 80)
    print()  # 空行
    sys.stdout.flush()
    
    all_keypoints = []
    all_labels = []
    
    success_count = 0
    fail_count = 0
    fail_list = []
    
    total = len(df)
    
    # ✅ 最簡單的迴圈，每 10 個顯示一次進度
    print("開始迴圈...")
    sys.stdout.flush()
    
    for i in range(total):
        # 每 10 個顯示進度
        if i % 10 == 0:
            print(f"進度: {i}/{total} ({i*100//total}%)")
            sys.stdout.flush()
        
        row = df.iloc[i]
        video_name = row['VIDEO_NAME']
        sentence = row['SENTENCE']
        
        video_path = os.path.join(videos_folder, video_name)
        
        if not os.path.exists(video_path):
            fail_count += 1
            fail_list.append((video_name, "檔案不存在"))
            continue
        
        try:
            keypoints = extract_hand_keypoints(video_path, max_frames)
            all_keypoints.append(keypoints)
            all_labels.append(sentence)
            success_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 使用者中斷")
            break
            
        except Exception as e:
            fail_count += 1
            fail_list.append((video_name, str(e)[:50]))
            continue
    
    print(f"\n進度: {total}/{total} (100%)")
    print("\n" + "=" * 80)
    print(f"處理完成！")
    print("=" * 80)
    print(f"✓ 成功: {success_count} 個")
    print(f"✗ 失敗: {fail_count} 個")
    sys.stdout.flush()
    
    if fail_count > 0 and len(fail_list) > 0:
        print(f"\n失敗清單 (前 10 個):")
        for video, reason in fail_list[:10]:
            print(f"  - {video}: {reason}")
        sys.stdout.flush()
    
    if len(all_keypoints) == 0:
        print("\n❌ 沒有成功處理任何影片！")
        return None, None
    
    # 儲存
    print("\n開始儲存資料...")
    sys.stdout.flush()
    
    all_keypoints = np.array(all_keypoints)
    all_labels = np.array(all_labels)
    
    keypoints_path = os.path.join(output_folder, 'keypoints.npy')
    labels_path = os.path.join(output_folder, 'labels.npy')
    
    np.save(keypoints_path, all_keypoints)
    np.save(labels_path, all_labels)
    
    print(f"\n" + "=" * 80)
    print(f"✓ 資料儲存成功")
    print("=" * 80)
    print(f"關鍵點形狀: {all_keypoints.shape}")
    print(f"標籤數量: {len(all_labels)}")
    print(f"\n儲存位置:")
    print(f"  - {keypoints_path}")
    print(f"  - {labels_path}")
    sys.stdout.flush()
    
    # 統計資訊
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\n詞彙統計 (共 {len(unique_labels)} 個詞彙):")
    
    for label, count in list(zip(unique_labels, counts))[:10]:
        print(f"  {label}: {count} 個樣本")
    
    if len(unique_labels) > 10:
        print(f"  ... (還有 {len(unique_labels) - 10} 個詞彙)")
    
    sys.stdout.flush()
    
    return all_keypoints, all_labels


if __name__ == "__main__":
    import time
    
    print("程式開始執行...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        keypoints, labels = process_dataset(
            excel_file='sign_language_dataset.xlsx',
            videos_folder='videos',
            output_folder='processed_data',
            max_frames=30
        )
        
        if keypoints is not None:
            elapsed_time = time.time() - start_time
            print(f"\n" + "=" * 80)
            print(f"🎉 全部完成！")
            print(f"   總耗時: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分鐘)")
            print("=" * 80)
        else:
            print("\n❌ 處理失敗")
    
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n程式結束")
        sys.stdout.flush()
