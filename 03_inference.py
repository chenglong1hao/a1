"""
手势识别推理系统 —— 场景隔离版

不同场景下仅识别该场景允许的手势：
- 普通模式：日常手势（求助、调温、烧水、开关灯等）
- 外卖事件：仅「外卖」「求助」
- 来访事件：仅「欢迎」「求助」

架构：GestureEngine（模型推理） + SceneManager（场景过滤 & 语义映射）
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

from config import (
    ACTIONS, SCENES, MODEL_PATH, SEQUENCE_LEN,
    CONF_THRESHOLD, STABLE_FRAMES, CAMERA_ID,
)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================
# 中文绘制
# =========================
def draw_chinese(img, text, pos, color=(255, 255, 255), size=32):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = None
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size)
            break
    if font is None:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# =========================
# 手势识别引擎
# =========================
class GestureEngine:
    """封装 MediaPipe + TensorFlow 模型，负责关键点提取与推理。"""

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.seq_buffer = deque(maxlen=SEQUENCE_LEN)

    def extract_keypoints(self, results):
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
            pts = pts - pts[0]
            return pts.flatten()
        return np.zeros(63, dtype=np.float32)

    def process_frame(self, frame):
        """处理一帧：绘制骨骼 + 提取关键点入队。返回 (annotated_frame, is_ready)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        kp = self.extract_keypoints(results)
        self.seq_buffer.append(kp)
        return frame, len(self.seq_buffer) == SEQUENCE_LEN

    def predict(self):
        """基于当前缓冲序列做一次推理，返回概率数组。"""
        seq = np.array(self.seq_buffer, dtype=np.float32)  # (T, 63)
        # 计算运动特征：位置 + 速度 + 加速度
        pos = seq
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
        feats = np.concatenate([pos, vel, acc], axis=-1)  # (T, 189)
        x = np.expand_dims(feats, axis=0)  # (1, T, 189)
        return self.model.predict(x, verbose=0)[0]

    def close(self):
        self.hands.close()


# =========================
# 场景管理器
# =========================
class SceneManager:
    """管理场景状态、手势过滤、稳定投票、语义映射。"""

    def __init__(self, initial="normal"):
        self.current = initial
        self.pred_buffer = deque(maxlen=STABLE_FRAMES)
        self.confirmed = False
        self.confirmed_at = 0
        self.feedback_msg = ""
        self.cooldown_until = 0
        self.last_label = None
        self.last_meaning = None
        self.last_conf = 0.0

    @property
    def in_cooldown(self):
        return time.time() < self.cooldown_until

    @property
    def cfg(self):
        return SCENES[self.current]

    @property
    def allowed(self):
        return self.cfg["gestures"]

    @property
    def name(self):
        return self.cfg["name"]

    def switch(self, scene_key):
        if scene_key not in SCENES:
            return False
        self.current = scene_key
        self.pred_buffer.clear()
        self.confirmed = False
        self.confirmed_at = 0
        self.feedback_msg = ""
        self.cooldown_until = 0
        self.last_label = None
        self.last_meaning = None
        self.last_conf = 0.0
        return True

    def gate(self, probs):
        """将不在本场景允许范围内的手势概率清零。"""
        gated = np.zeros_like(probs, dtype=np.float32)
        idxs = [ACTIONS.index(g) for g in self.allowed]
        gated[idxs] = probs[idxs]
        return gated

    def update(self, probs):
        """
        输入原始概率，输出 (label, meaning, confidence, is_stable)。

        流程：概率门控 → 置信度过滤 → 稳定帧投票 → 语义映射。
        仅在 is_stable=True 时 meaning 有效。
        """
        gated = self.gate(probs)
        best_idx = int(np.argmax(gated))
        best_conf = float(gated[best_idx])
        best_label = ACTIONS[best_idx]

        if best_conf < CONF_THRESHOLD:
            self.pred_buffer.clear()
            return best_label, None, best_conf, False

        self.pred_buffer.append(best_label)

        if len(self.pred_buffer) < STABLE_FRAMES:
            return best_label, None, best_conf, False

        final_label, count = Counter(self.pred_buffer).most_common(1)[0]
        if count < STABLE_FRAMES - 1:
            return best_label, None, best_conf, False

        meaning = self.cfg["meanings"].get(final_label, final_label)
        # "无动作"不触发冷却，只有有意义的手势才锁定3s
        if final_label != "无动作":
            self.cooldown_until = time.time() + 3.0
            self.last_label = final_label
            self.last_meaning = meaning
            self.last_conf = best_conf
        return final_label, meaning, best_conf, True


# =========================
# 主循环
# =========================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在: {MODEL_PATH}")
        return

    engine = GestureEngine(MODEL_PATH)
    scene = SceneManager("normal")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("摄像头打开失败")
        engine.close()
        return

    WIN = "手势识别 · 场景模式"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1400, 900)

    status_text = "系统待命中..."
    status_color = (0, 255, 255)
    was_cooldown = False

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame, ready = engine.process_frame(frame)

        # --- 推理 & 场景过滤 ---
        if ready and not scene.in_cooldown:
            # 刚从冷却退出，清空序列缓冲避免重复识别
            if was_cooldown:
                engine.seq_buffer.clear()
                was_cooldown = False
                if scene.current != "normal" and scene.confirmed:
                    scene.switch("normal")
                    status_text = "已自动返回普通模式"
                    status_color = (0, 255, 255)
                else:
                    scene.pred_buffer.clear()
                    status_text = "识别中..."
                    status_color = (255, 200, 0)
            else:
                probs = engine.predict()
                label, meaning, conf, stable = scene.update(probs)

                if stable and meaning:
                    status_text = f"识别：{meaning}（{label}，{conf:.2f}）"
                    status_color = (0, 255, 0)

                    # 事件模式：确认 / 取消 反馈
                    if scene.current != "normal" and not scene.confirmed:
                        primary = scene.cfg["gestures"][0]
                        if label == primary:
                            scene.feedback_msg = scene.cfg.get("confirm_msg", "")
                            scene.confirmed = True
                            scene.confirmed_at = time.time()
                        elif label == "取消":
                            scene.feedback_msg = scene.cfg.get("cancel_msg", "")
                            scene.confirmed = True
                            scene.confirmed_at = time.time()
                elif label:
                    status_text = f"识别中：{label}（{conf:.2f}）"
                    status_color = (255, 200, 0)
                else:
                    status_text = "识别中..."
                    status_color = (255, 200, 0)

        elif scene.in_cooldown:
            was_cooldown = True
            remaining = scene.cooldown_until - time.time()
            status_text = f"已识别：{scene.last_meaning}（{remaining:.1f}s 后继续）"
            status_color = (0, 255, 0)

        # --- UI 绘制 ---
        h, w = frame.shape[:2]
        panel_h = 195
        cv2.rectangle(frame, (0, 0), (w, panel_h), (30, 30, 30), -1)

        y = 12
        frame = draw_chinese(frame, status_text, (20, y), status_color, 33)
        y += 48
        frame = draw_chinese(
            frame,
            f"当前场景：{scene.name}",
            (20, y), (220, 220, 220), 26,
        )
        y += 34
        allowed_str = "可识别手势：" + " · ".join(scene.allowed)
        frame = draw_chinese(frame, allowed_str, (20, y), (160, 200, 255), 24)

        if scene.feedback_msg:
            y += 38
            frame = draw_chinese(frame, scene.feedback_msg, (20, y), (100, 255, 200), 28)

        cv2.imshow(WIN, frame)

        # --- 键盘 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("0"):
            scene.switch("normal")
            engine.seq_buffer.clear()
            status_text = "已切换到普通模式"
            status_color = (0, 255, 255)
        elif key == ord("1"):
            scene.switch("takeaway")
            engine.seq_buffer.clear()
            status_text = "已切换到外卖事件模式"
            status_color = (255, 220, 120)
            scene.feedback_msg = SCENES["takeaway"].get("trigger_msg", "")
        elif key == ord("2"):
            scene.switch("visit")
            engine.seq_buffer.clear()
            status_text = "已切换到来访事件模式"
            status_color = (120, 220, 255)
            scene.feedback_msg = SCENES["visit"].get("trigger_msg", "")

    cap.release()
    cv2.destroyAllWindows()
    engine.close()


if __name__ == "__main__":
    main()
