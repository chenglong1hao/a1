import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import ACTIONS, DATA_DIR, NUM_SEQUENCES, SEQUENCE_LEN, MODEL_PATH


def compute_motion_features(sequence):
    """
    sequence: (T, 63) 原始关键点
    returns:  (T, 189) = 位置 + 速度 + 加速度
    """
    T = sequence.shape[0]
    pos = sequence
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([pos, vel, acc], axis=-1)


def load_data():
    X, y = [], []
    label_map = {a: i for i, a in enumerate(ACTIONS)}

    for action in ACTIONS:
        for seq in range(NUM_SEQUENCES):
            frames = []
            seq_dir = os.path.join(DATA_DIR, action, str(seq))
            for f in range(SEQUENCE_LEN):
                p = os.path.join(seq_dir, f"{f}.npy")
                frames.append(np.load(p))
            X.append(frames)
            y.append(label_map[action])

    X = np.array(X, dtype=np.float32)  # [N, T, 63]
    X = np.array([compute_motion_features(seq) for seq in X], dtype=np.float32)  # [N, T, 189]
    y = to_categorical(y, num_classes=len(ACTIONS))
    return X, y


def build_model(num_classes):
    model = Sequential([
        Input(shape=(SEQUENCE_LEN, 189)),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = build_model(len(ACTIONS))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max")
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=8,
        callbacks=callbacks
    )

    print(f"训练完成，最佳模型已保存：{MODEL_PATH}")


if __name__ == "__main__":
    main()
