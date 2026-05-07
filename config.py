import os

# =========================
# 1) 训练 & 采集参数
# =========================
ACTIONS = ["外卖", "欢迎", "求助", "降温", "升温", "烧水", "开灯", "关灯", "无动作","取消"]

ACTION_ZH = {
    "取消": "取消",
    "外卖": "外卖",
    "欢迎": "欢迎",
    "求助": "求助",
    "降温": "降温",
    "升温": "升温",
    "烧水": "烧水",
    "开灯": "开灯",
    "关灯": "关灯",
    "无动作": "无动作",
}

DATA_DIR = "gesture_data"
MODEL_PATH = "gesture_model.h5"

NUM_SEQUENCES = 15
SEQUENCE_LEN = 30
CONF_THRESHOLD = 0.70
STABLE_FRAMES = 6
CAMERA_ID = 0

# =========================
# 2) 场景定义
# =========================
# 每个场景定义：
#   name         - 场景显示名称
#   gestures     - 该场景下可识别的手势集合（ACTIONS 的子集）
#   meanings     - 手势 → 语义文本映射
#   trigger_msg  - 进入场景时的提示（可选）
#   confirm_msg  - 确认手势后的反馈（可选）
#   cancel_msg   - 取消手势后的反馈（可选）

SCENES = {
    "normal": {
        "name": "普通模式",
        "gestures": ["求助", "降温", "升温", "烧水", "开灯", "关灯", "无动作"],
        "meanings": {
            "求助": "紧急求助",
            "降温": "空调调低温度",
            "升温": "空调调高温度",
            "烧水": "开始烧水",
            "开灯": "打开灯光",
            "关灯": "关闭灯光",
            "无动作": "待命",
        },
    },
    "takeaway": {
        "name": "外卖事件",
        "gestures": ["外卖", "取消"],
        "meanings": {
            "外卖": "确认：门外有外卖",
            "取消": "取消本次外卖确认",
        },
        "trigger_msg": "【传感器】门口检测到疑似外卖员，请做「外卖」手势确认，或取消。",
        "confirm_msg": "已确认：门外有外卖，请及时取餐。",
        "cancel_msg": "已取消外卖确认。",
    },
    "visit": {
        "name": "来访事件",
        "gestures": ["欢迎", "取消"],
        "meanings": {
            "欢迎": "确认：欢迎来访",
            "取消": "取消本次来访确认",
        },
        "trigger_msg": "【传感器】门口检测到来访人员，请做「欢迎」手势确认，或拒绝。",
        "confirm_msg": "已确认：有人来访，请注意查看。",
        "cancel_msg": "已取消来访确认。",
    },
}

# =========================
# 3) 配置校验
# =========================
for _scene_key, _scene in SCENES.items():
    for _g in _scene["gestures"]:
        assert _g in ACTIONS, f"场景 [{_scene_key}] 中的手势 '{_g}' 不在 ACTIONS 中"
    for _g in _scene["meanings"]:
        assert _g in _scene["gestures"], f"场景 [{_scene_key}] meanings 中的手势 '{_g}' 不在该场景 gestures 中"
