import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from collections import Counter

# 載入資料
print("載入資料...")
keypoints = np.load('processed_data/keypoints.npy')
labels = np.load('processed_data/labels.npy')

print(f"原始關鍵點形狀: {keypoints.shape}")
print(f"原始標籤數量: {len(labels)}")

# 統計每個標籤出現的次數
label_counts = Counter(labels)
print(f"\n標籤統計:")
print(f"  總共 {len(label_counts)} 個不同的句子")
print(f"  只出現 1 次的句子: {sum(1 for c in label_counts.values() if c == 1)} 個")
print(f"  出現 2 次的句子: {sum(1 for c in label_counts.values() if c == 2)} 個")
print(f"  出現 ≥ 3 次的句子: {sum(1 for c in label_counts.values() if c >= 3)} 個")
print(f"  出現 ≥ 4 次的句子: {sum(1 for c in label_counts.values() if c >= 4)} 個")

# ✅ 修正：只保留出現次數 >= 3 的類別
min_samples = 3
valid_labels = [label for label, count in label_counts.items() if count >= min_samples]
valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]

keypoints_filtered = keypoints[valid_indices]
labels_filtered = labels[valid_indices]

print(f"\n過濾後 (min_samples={min_samples}):")
print(f"  保留樣本數: {len(keypoints_filtered)}")
print(f"  保留類別數: {len(set(labels_filtered))}")

# 檢查是否有足夠的資料
num_unique_classes = len(set(labels_filtered))
num_samples = len(keypoints_filtered)
test_size_ratio = 0.2
estimated_test_size = int(num_samples * test_size_ratio)

print(f"\n可行性檢查:")
print(f"  預計測試集大小: {estimated_test_size}")
print(f"  類別數: {num_unique_classes}")

if estimated_test_size < num_unique_classes:
    print(f"  ⚠️ 測試集太小！調整 test_size...")
    # 計算最小的 test_size
    min_test_size = num_unique_classes / num_samples
    test_size_ratio = max(min_test_size * 1.2, 0.25)  # 至少 25%
    print(f"  新的 test_size: {test_size_ratio:.2f}")

# 編碼標籤
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_filtered)
num_classes = len(label_encoder.classes_)

print(f"\n最終詞彙類別數: {num_classes}")
print(f"詞彙列表 (前 20 個): {label_encoder.classes_[:20]}")

# 顯示每個類別的樣本數統計
class_counts = Counter(labels_filtered)
counts_distribution = Counter(class_counts.values())
print(f"\n每個類別的樣本數分布:")
for count, num_classes_with_count in sorted(counts_distribution.items()):
    print(f"  {count} 個樣本: {num_classes_with_count} 個類別")

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    keypoints_filtered, 
    labels_encoded, 
    test_size=test_size_ratio,
    random_state=42,
    stratify=labels_encoded
)

print(f"\n訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 建立 LSTM 模型
def create_lstm_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # LSTM 層
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.4),
        
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.4),
        
        layers.LSTM(32),
        layers.Dropout(0.4),
        
        # 全連接層
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        
        # 輸出層
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 建立模型
input_shape = (keypoints_filtered.shape[1], keypoints_filtered.shape[2])
model = create_lstm_model(input_shape, num_classes)

# 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 訓練模型
print("\n開始訓練...")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=8,
            factor=0.5,
            verbose=1
        )
    ],
    verbose=1
)

# 評估模型
print("\n評估模型...")
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"\n{'='*60}")
print(f"🎯 最終結果")
print(f"{'='*60}")
print(f"測試準確率: {test_acc:.2%}")
print(f"測試損失: {test_loss:.4f}")
print(f"訓練 Epochs: {len(history.history['loss'])}")
print(f"最佳驗證準確率: {max(history.history['val_accuracy']):.2%}")
print(f"{'='*60}")

# 儲存模型
os.makedirs('processed_data/model', exist_ok=True)
model.save('processed_data/model/sign_language_lstm.keras')
np.save('processed_data/model/label_encoder.npy', label_encoder.classes_)

print("\n✓ 模型已儲存至 processed_data/model/")

# 繪製訓練曲線
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.title('Model Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.title('Model Loss', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('processed_data/model/training_history.png', dpi=150, bbox_inches='tight')
print("✓ 訓練曲線已儲存")

# Top-5 準確率（更寬鬆的指標）
print("\n計算 Top-5 準確率...")
y_pred_proba = model.predict(X_test, verbose=0)
top5_predictions = np.argsort(y_pred_proba, axis=1)[:, -5:]  # 取前 5 個預測
top5_accuracy = np.mean([y_test[i] in top5_predictions[i] for i in range(len(y_test))])
print(f"Top-5 準確率: {top5_accuracy:.2%}")