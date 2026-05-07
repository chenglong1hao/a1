import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from config import ACTIONS, DATA_DIR, NUM_SEQUENCES, SEQUENCE_LEN, CAMERA_ID, ACTION_ZH

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# =========================
# 中文绘制工具（PIL）
# =========================
def cv2_add_chinese_text(img, text, position, text_color=(255, 255, 255), text_size=30):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img

    draw = ImageDraw.Draw(img_pil)
    font_path_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc"
    ]

    font = None
    for fp in font_path_candidates:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, text_size)
            break
    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =========================
# 关键点提取
# =========================
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        pts = pts - pts[0]
        return pts.flatten()
    return np.zeros(63, dtype=np.float32)


# =========================
# 目录相关
# =========================
def ensure_action_dir(action):
    os.makedirs(os.path.join(DATA_DIR, action), exist_ok=True)

def get_next_sequence_index(action):
    """返回该动作下下一个可用序号，避免覆盖旧数据"""
    action_dir = os.path.join(DATA_DIR, action)
    if not os.path.exists(action_dir):
        return 0
    seq_dirs = []
    for name in os.listdir(action_dir):
        if name.isdigit():
            seq_dirs.append(int(name))
    if len(seq_dirs) == 0:
        return 0
    return max(seq_dirs) + 1


# =========================
# 交互：选择动作
# =========================
def choose_actions():
    print("\n====== 动作列表 ======")
    for i, a in enumerate(ACTIONS, start=1):
        print(f"{i}. {ACTION_ZH.get(a, a)} ")

    print("\n输入要录制的动作编号：")
    print(" - 单个")
    print(" - 多个")
    print(" - 全部(all)")
    raw = input("你的选择: ").strip().lower()

    if raw == "all":
        selected = ACTIONS[:]
    else:
        idxs = []
        for x in raw.split(","):
            x = x.strip()
            if x.isdigit():
                idx = int(x)
                if 1 <= idx <= len(ACTIONS):
                    idxs.append(idx - 1)
        idxs = sorted(list(set(idxs)))
        selected = [ACTIONS[i] for i in idxs]

    if len(selected) == 0:
        print("未选择有效动作，默认选择全部动作。")
        selected = ACTIONS[:]

    return selected



# =========================
# 主流程
# =========================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    selected_actions = choose_actions()
    plan = []
    print("\n====== 本次录制计划 ======")
    for a in selected_actions:
        plan.append((a, NUM_SEQUENCES))
        print(f"- {ACTION_ZH.get(a, a)}: {NUM_SEQUENCES} 组")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("摄像头打开失败，请检查 CAMERA_ID 或摄像头占用。")
        return

    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        for i, (action, need_sequences) in enumerate(plan):
            action_zh = ACTION_ZH.get(action, action)
            ensure_action_dir(action)

            start_seq = 0  # 每次重新录制从0开始，覆盖旧数据
            end_seq = need_sequences

            print(f"\n===== 当前动作：{action_zh} ({action}) =====")
            print(f"将录制 {need_sequences} 组，每组 {SEQUENCE_LEN} 帧")
            print("按 空格 开始；按 q 退出")

            # 动作开始前确认
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                frame = cv2_add_chinese_text(frame, f"当前手势：{action_zh}", (20, 20), (255, 255, 0), 34)
                frame = cv2_add_chinese_text(frame, f"本次录制 {need_sequences} 组（每组 {SEQUENCE_LEN} 帧）", (20, 70), (180, 255, 180), 28)
                frame = cv2_add_chinese_text(frame, "按 空格 开始，按 Q 退出", (20, 145), (180, 255, 180), 28)

                cv2.imshow("Collect Data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # --- 3-2-1 倒计时（每个手势仅一次）---
            for count in [3, 2, 1]:
                start_tick = cv2.getTickCount()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    if results.multi_hand_landmarks:
                        for hand_lm in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                    frame = cv2_add_chinese_text(frame, f"准备录制：{action_zh}", (20, 20), (255, 255, 0), 34)
                    frame = cv2_add_chinese_text(frame, f"共 {need_sequences} 组，每组 {SEQUENCE_LEN} 帧", (20, 70), (255, 255, 0), 28)
                    frame = cv2_add_chinese_text(frame, f"倒计时：{count}", (20, 140), (0, 255, 255), 60)

                    cv2.imshow("Collect Data", frame)
                    elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
                    if elapsed >= 1.0:
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            # 开始连续录制所有组
            for seq in range(start_seq, end_seq):
                seq_dir = os.path.join(DATA_DIR, action, str(seq))
                os.makedirs(seq_dir, exist_ok=True)

                done_seq = seq - start_seq + 1

                for f in range(SEQUENCE_LEN):
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        for hand_lm in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(seq_dir, f"{f}.npy"), keypoints)

                    frame = cv2_add_chinese_text(frame, f"正在录制：{action_zh}", (20, 20), (0, 255, 0), 34)
                    frame = cv2_add_chinese_text(
                        frame,
                        f"第 {done_seq}/{need_sequences} 组，第 {f + 1}/{SEQUENCE_LEN} 帧",
                        (20, 70),
                        (0, 220, 0),
                        28
                    )
                    frame = cv2_add_chinese_text(frame, "请持续做动作！按 Q 可提前退出", (20, 115), (220, 220, 220), 26)

                    cv2.imshow("Collect Data", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            print(f"动作 [{action_zh}] 录制完成。")

            # 动作间切换提示
            if i < len(plan) - 1:
                next_action = plan[i + 1][0]
                next_action_zh = ACTION_ZH.get(next_action, next_action)
                print(f"按空格进入下一个动作：{next_action_zh}")

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    frame = cv2_add_chinese_text(frame, f"动作【{action_zh}】录制完成", (20, 20), (255, 200, 0), 34)
                    frame = cv2_add_chinese_text(frame, f"下一个：{next_action_zh}", (20, 70), (255, 255, 0), 30)
                    frame = cv2_add_chinese_text(frame, "按 空格 继续，按 Q 退出", (20, 115), (180, 255, 180), 28)

                    cv2.imshow("Collect Data", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        break
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("本次选择的动作全部录制完成！")


if __name__ == "__main__":
    main()
