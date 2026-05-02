"""
情绪分析引擎 —— 关键词触发 + LLM分析 + 情绪感染映射 + 突变检测
"""
import re
import json
import random
from langchain_core.messages import HumanMessage, SystemMessage

# ==================== 可配置常量（集中管理于 config.py）====================
from config import (
    EMOTION_HALF_LIFE, EMOTION_BASELINE,
    RAPID_DECAY_FACTOR,
    DELTA_SCALE, DELTA_MAX,
    FORGETTING_PROBABILITY, ANCHOR_RESURRECT_FACTOR,
)

# ==================== 关键词触发 ====================
# (关键词, 情绪, 强度, 置信度)
TRIGGER_WORDS = [
    # 单字/符号（硬编码，避免浪费 LLM 调用）
    # 注意：单字匹配优先级高，匹配后不再走 LLM 情绪分析
    ("？", "惊讶", 2, 0.5),     # 疑惑
    ("？？", "惊讶", 3, 0.6),
    ("？？？", "惊讶", 4, 0.7),
    ("。。", "悲伤", 3, 0.6),
    ("。。。", "悲伤", 4, 0.7),
    ("……", "悲伤", 3, 0.6),
    ("6", "快乐", 4, 0.7),
    ("666", "快乐", 6, 0.8),
    # 注意："好""啊""哦""嗯"等太常见，不在这里匹配，避免误触发
    # 愤怒
    ("菜", "愤怒", 70, 0.9),
    ("笨", "愤怒", 60, 0.8),
    ("傻", "愤怒", 50, 0.7),
    ("滚", "愤怒", 80, 0.95),
    ("烦", "愤怒", 50, 0.7),
    ("闭嘴", "愤怒", 80, 0.9),
    ("无语", "愤怒", 40, 0.6),
    ("傻逼", "愤怒", 80, 0.95),
    ("sb", "愤怒", 70, 0.9),
    ("畜生", "愤怒", 90, 0.95),
    ("cs", "愤怒", 70, 0.85),
    ("废物", "愤怒", 80, 0.9),
    # 快乐
    ("哈哈", "快乐", 60, 0.8),
    ("笑死", "快乐", 70, 0.85),
    ("好玩", "快乐", 50, 0.7),
    ("厉害", "快乐", 50, 0.7),
    ("笑死我了", "快乐", 80, 0.9),
    ("可爱", "快乐", 60, 0.8),
    ("好萌", "快乐", 70, 0.85),
    ("乖", "快乐", 40, 0.7),
    ("萌萌", "快乐", 60, 0.8),
    ("抱抱", "快乐", 60, 0.8),
    ("贴贴", "快乐", 70, 0.85),
    ("想你", "快乐", 50, 0.7),
    ("想你了", "快乐", 60, 0.8),
    # 悲伤
    ("委屈", "悲伤", 70, 0.9),
    ("难过", "悲伤", 60, 0.8),
    ("伤心", "悲伤", 70, 0.85),
    # 恐惧
    ("吃醋", "恐惧", 70, 0.9),
]


def check_trigger_words(text: str) -> list[tuple[str, int, float]]:
    """命中关键词返回所有匹配情绪 [(emotion, intensity, confidence), ...]，否则空列表"""
    results = []
    for word, emotion, intensity, confidence in TRIGGER_WORDS:
        if word in text:
            results.append((emotion, intensity, confidence))
    # 去重：同一情绪只保留第一次命中
    seen = {}
    for e, i, c in results:
        if e not in seen and e != "中性":
            seen[e] = (e, i, c)
    return list(seen.values())


# ==================== 情绪感染映射 ====================

# 基础变化量（实际 delta = min(基础量 × intensity × confidence × DELTA_SCALE, DELTA_MAX)）
INFECTION_UPDATES = {
    ("快乐", "夸AI"): {"快乐": 2, "惊讶": 1},
    ("快乐", "分享好事"): {"快乐": 3},
    ("愤怒", "对AI不满"): {"悲伤": 2, "愤怒": 1},
    ("愤怒", "对其他人"): {"愤怒": 1},
    ("悲伤", None): {"悲伤": 1, "快乐": -1},
    ("恐惧", None): {"恐惧": 2, "快乐": -1},
    ("惊讶", None): {"惊讶": 2, "快乐": 1},
    ("厌恶", None): {"厌恶": 2, "愤怒": 1},
    ("中性", None): {},
}

# 上下文分类规则
CONTEXT_RULES = {
    "快乐": {
        "夸AI": lambda t: any(w in t for w in ["你", "你真", "你好", "你太", "你好可爱", "你好萌"]),
        "分享好事": lambda t: True,  # 兜底
    },
    "愤怒": {
        "对AI不满": lambda t: any(w in t for w in ["你", "你才", "你是个", "你不行", "你菜"]),
        "对其他人": lambda t: True,  # 兜底
    },
}

# 额外情境规则
EXTRA_RULES = {
    # 用户快乐提到别人 → 恐惧 +1（吃醋替代）
    ("快乐", "提到别人"): lambda text: any(w in text for w in ["他", "她", "别人", "同事", "朋友", "同学", "老板"]),
}

# 好感度变化系数（每强度单位的变化量，范围 0~10000）
# 目标：单次对话 10~100 点（0.1%~1%），极端走心 200~500 点（2%~5%）
AFFECTION_PER_INTENSITY = {
    "快乐": 5,     # 强度5 → +25
    "惊讶": 3,     # 强度5 → +15
    "悲伤": -5,    # 强度5 → -25
    "愤怒": -8,    # 强度5 → -40
    "恐惧": -6,    # 强度5 → -30
    "厌恶": -6,    # 强度5 → -30
}


# ==================== 反转信号（突变检测） ====================

SHIFT_SIGNALS = {
    "愤怒": {
        "keywords": ["笑死", "哈哈", "好吧算了", "噗", "服了", "算了"],
        "target": "快乐",
    },
    "悲伤": {
        "keywords": ["其实还好", "也没那么糟", "我开玩笑的", "没事啦"],
        "target": "快乐",
    },
}

# ==================== 核心函数 ====================

def rapid_decay(emotions: dict, factor: float = RAPID_DECAY_FACTOR) -> dict:
    """快速衰减层：所有情绪乘以衰减因子"""
    if not emotions:
        return {}
    result = {}
    for k, v in emotions.items():
        new_v = v * factor
        if new_v > 0.01:
            result[k] = round(new_v, 2)
    return result


def apply_forgetting(emotions: dict, baseline: dict = None, protected_labels: set = None) -> dict:
    """
    遗忘机制：强度≤2 且高于基底的情绪，默认 20% 概率归零。
    protected_labels 中的标签降低遗忘概率至 5%。
    """
    if not emotions:
        return {}
    baseline = baseline or {}
    protected_labels = protected_labels or set()
    result = dict(emotions)
    for k in list(result.keys()):
        bv = baseline.get(k, 0)
        if bv < result[k] <= 20:
            prob = FORGETTING_PROBABILITY
            if k in protected_labels:
                prob = 0.05  # 有锚点的标签遗忘概率降至5%
            if random.random() < prob:
                if bv > 0:
                    result[k] = bv
                else:
                    del result[k]
    return result


LLM_ANALYSIS_PROMPT = """你是一个情绪分析助手。分析用户消息中的情绪，从以下 6 个基础情绪标签中选择一个或多个，外加"中性"：

- 快乐：高兴、喜悦、开心、分享好事、开玩笑、调侃、吐槽、被夸、感到温暖
- 悲伤：难过、伤心、失落、委屈、被误解、受到不公对待、孤独
- 愤怒：生气、不满、烦躁、暴躁、咬牙切齿
- 恐惧：害怕、不安、焦虑、担忧、吃醋、缺乏安全感
- 惊讶：震惊、意外、出乎意料、困惑、疑惑
- 厌恶：讨厌、恶心、反感、看不惯
- 中性：无明显情绪、日常陈述、客观描述、无奈

区分要点：
- 调侃/吐槽 → 快乐或中性（除非有明确恶意）
- 被误解感到委屈 → 悲伤；但纠正/解释/争论 → 中性、愤怒或快乐
- 吃醋/嫉妒 → 恐惧（害怕失去）
- 无奈/接受 → 中性
- 一条消息可能包含多种情绪，请全部列出。中性不应与其他情绪同时出现。

<examples>
用户消息：我今天升职了！
输出：{{"emotion": "快乐", "intensity": 80, "confidence": 0.95}}

用户消息：你好可爱啊
输出：{{"emotion": "快乐", "intensity": 60, "confidence": 0.8}}  ← 被夸感到快乐

用户消息：你真是个笨蛋
输出：{{"emotion": "愤怒", "intensity": 50, "confidence": 0.7}}

用户消息：最近天气不错
输出：{{"emotion": "中性", "intensity": 10, "confidence": 0.6}}

用户消息：我好委屈啊
输出：{{"emotion": "悲伤", "intensity": 80, "confidence": 0.9}}

用户消息：抱抱我嘛~
输出：{{"emotion": "快乐", "intensity": 70, "confidence": 0.85}}  ← 撒娇求关注→快乐

用户消息：你今天和别人聊得挺开心的嘛
输出：{{"emotion": "恐惧", "intensity": 60, "confidence": 0.8}}  ← 吃醋→害怕失去

用户消息：这个图里明明有闪电啊，你再仔细看看
输出：{{"emotion": "中性", "intensity": 20, "confidence": 0.7}}  ← 解释/指正，无负面情绪

用户消息：你理解错啦，我是说那只猪在飞呢
输出：{{"emotions": [{{"emotion": "快乐", "intensity": 40, "confidence": 0.7}}]}}  ← 调侃语气

用户消息：哈哈你好可爱哦
输出：{{"emotions": [{{"emotion": "快乐", "intensity": 60, "confidence": 0.8}}, {{"emotion": "惊讶", "intensity": 30, "confidence": 0.6}}]}}

用户消息：可爱死了笑死我了
输出：{{"emotions": [{{"emotion": "快乐", "intensity": 70, "confidence": 0.85}}, {{"emotion": "惊讶", "intensity": 40, "confidence": 0.7}}]}}

用户消息：我好害怕你不理我
输出：{{"emotion": "恐惧", "intensity": 80, "confidence": 0.9}}
</examples>

最近对话上下文（供参考）：
{context}

用户消息：{text}

请只输出 JSON，不要包含其他内容：
{{"emotions": [{{"emotion": "标签", "intensity": 1-100, "confidence": 0-1}}, ...]}}"""


async def analyze_user_emotion(text: str, context_messages: list, llm) -> list[tuple[str, int, float]]:
    """
    分析用户情绪（多标签）。
    返回 [(emotion, intensity, confidence), ...]，失败返回 []。
    """
    # 1. 关键词优先触发（可能命中多个）
    trigger_results = check_trigger_words(text)
    if trigger_results:
        print(f"[情绪] 关键词触发: {trigger_results}")
        return trigger_results

    # 2. LLM 分析
    context_str = ""
    if context_messages:
        recent = context_messages[-3:]
        context_lines = []
        for msg in recent:
            if hasattr(msg, "content"):
                context_lines.append(str(msg.content)[:80])
            else:
                context_lines.append(str(msg)[:80])
        context_str = "\n".join(context_lines)

    try:
        prompt = LLM_ANALYSIS_PROMPT.format(context=context_str, text=text)
        messages = [
            SystemMessage(content="你是一个情绪分析助手。只输出JSON。"),
            HumanMessage(content=prompt),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip()
        # 提取完整 JSON（匹配最外层 {}，支持嵌套结构）
        json_start = content.find("{")
        json_end = content.rfind("}")
        if json_start != -1 and json_end > json_start:
            data = json.loads(content[json_start:json_end + 1])
            raw_list = data.get("emotions", [])
            result = []
            valid_emotions = {"快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"}
            for item in raw_list:
                emotion = item.get("emotion", "中性")
                if emotion not in valid_emotions:
                    continue
                intensity = max(1, min(100, int(item.get("intensity", 5))))
                confidence = max(0, min(1, float(item.get("confidence", 0.5))))
                if confidence < 0.5:
                    continue
                result.append((emotion, intensity, confidence))
            # 去重
            seen = {}
            for e, i, c in result:
                if e not in seen and e != "中性":
                    seen[e] = (e, i, c)
            final = list(seen.values())
            if final:
                print(f"[情绪] LLM分析: {final}")
                return final

        print(f"[情绪] LLM输出解析失败: {content[:100]}")
    except Exception as e:
        print(f"[情绪] LLM调用失败: {e}")

    # 3. 兜底：关键词规则
    print(f"[情绪] 使用关键词兜底")
    fallback_map = {
        "菜": ("愤怒", 5, 0.5), "笨": ("愤怒", 5, 0.5),
        "哈哈": ("快乐", 4, 0.5), "可爱": ("快乐", 4, 0.5),
        "抱抱": ("快乐", 4, 0.5),
    }
    for word, result in fallback_map.items():
        if word in text:
            return [result]
    print(f"[警告] 情绪分析完全失败，LLM 无输出且无关键词命中: {text[:80]}")
    return []


def classify_user_context(user_emotion: str, text: str) -> str:
    """判断用户情绪的上下文（开心→夸AI/分享好事，生气→对AI不满/对其他人）"""
    if user_emotion not in CONTEXT_RULES:
        return None
    for ctx, test_fn in CONTEXT_RULES[user_emotion].items():
        if test_fn(text):
            return ctx
    return None


# ==================== 和解映射 ====================

RECONCILIATION_RULES = [
    # (关键词列表, 正向匹配, 效果: {emotion: delta}, 悲伤≥7时替代效果)
    (["对不起", "我错了", "别生气"], True,
     {"愤怒": -3, "悲伤": -3, "快乐": 1},
     {"愤怒": -3, "悲伤": -1, "快乐": 1}),
    (["不气不气", "乖"], True,
     {"愤怒": -2, "快乐": 2},
     None),
    (["开玩笑的", "逗你的"], True,
     {"悲伤": -3, "愤怒": 1},
     {"悲伤": -1, "愤怒": 1}),
]


def check_reconciliation(text: str, current_emotions: dict) -> dict:
    """
    检查用户文本是否包含和解关键词。
    命中返回情绪变化 dict，否则返回 None。
    如果当前悲伤 >= 7，使用替代效果（减半悲伤下降量）。
    """
    for keywords, _, base_updates, alt_updates in RECONCILIATION_RULES:
        for kw in keywords:
            if kw in text:
                current_beishang = current_emotions.get("悲伤", 0)
                if current_beishang >= 7 and alt_updates is not None:
                    print(f"[和解] 关键词'{kw}'命中，悲伤≥7，使用替代效果: {alt_updates}")
                    return dict(alt_updates)
                print(f"[和解] 关键词'{kw}'命中，基础效果: {base_updates}")
                return dict(base_updates)
    return None


def compute_infection_updates(user_emotions: list[tuple[str, int, float]],
                              text: str, current_emotions: dict) -> dict:
    """
    根据情绪感染映射计算 AI 情绪变化量（多标签版本）。
    user_emotions: [(emotion, intensity, confidence), ...]
    返回 {emotion: delta, ...}
    """
    updates = {}

    # 对每种检测到的情绪分别计算感染
    for emotion, intensity, confidence in user_emotions:
        if emotion == "中性" or confidence < 0.5:
            continue

        context = classify_user_context(emotion, text)
        key = (emotion, context)
        if key not in INFECTION_UPDATES:
            key = (emotion, None)
        base_updates = INFECTION_UPDATES.get(key, {})

        for target_emo, base_delta in base_updates.items():
            if base_delta < 0:
                updates[target_emo] = updates.get(target_emo, 0) + base_delta
            else:
                scaled = base_delta * intensity * confidence * DELTA_SCALE
                scaled = min(scaled, DELTA_MAX)
                updates[target_emo] = updates.get(target_emo, 0) + round(scaled, 1)

    # ===== 和解映射（与正常感染合并，覆盖冲突项） =====
    reconciliation_updates = check_reconciliation(text, current_emotions)
    if reconciliation_updates is not None:
        for k, v in reconciliation_updates.items():
            updates[k] = v
        print(f"[情绪] 和解覆盖更新: {updates}")
        return updates

    # ===== 额外规则（仅非和解时生效） =====
    emotion_labels = [e for e, _, _ in user_emotions]

    # 1. 用户快乐且提到别人 → 恐惧 +1（吃醋替代）
    if "快乐" in emotion_labels and EXTRA_RULES[("快乐", "提到别人")](text):
        updates["恐惧"] = updates.get("恐惧", 0) + 1

    # 2. 用户愤怒（对AI不满）且 AI 当前有恐惧 → 恐惧 +2（越凶越不安）
    if "愤怒" in emotion_labels:
        ctx = classify_user_context("愤怒", text)
        if ctx == "对AI不满" and current_emotions.get("恐惧", 0) > 0:
            updates["恐惧"] = updates.get("恐惧", 0) + 2

    # 3. 反转抑制：AI 愤怒 > 5 且用户示好(快乐) → 愤怒额外 -3
    if current_emotions.get("愤怒", 0) > 5 and "快乐" in emotion_labels:
        updates["愤怒"] = updates.get("愤怒", 0) - 3

    return updates


def get_trigger_reason(user_emotion: str, text: str) -> str:
    """根据用户情绪生成触发原因文本"""
    context = classify_user_context(user_emotion, text)
    reasons = {
        ("快乐", "夸AI"): "被夸了",
        ("快乐", "分享好事"): "用户分享了开心的事",
        ("愤怒", "对AI不满"): "被用户凶了",
        ("愤怒", "对其他人"): "用户在对别人生气",
        "悲伤": "用户在悲伤",
        "恐惧": "用户没有安全感",
        "惊讶": "用户很惊讶",
        "厌恶": "用户表示厌恶",
    }
    return reasons.get((user_emotion, context)) or reasons.get(user_emotion, "")


def compute_affection_delta(emotion_dict: dict, current_affection: float = 0) -> int:
    """
    根据 AI 当前情绪向量计算好感度变化量。
    每 intensity 单位 × 系数，各情绪累加。
    例如：撒娇=5 → 5×200=+1000，生气=3 → 3×(-200)=-600

    好感度软上限：正向增长时，好感度越高增量越小。
    在 5000/10000 时增益减半，8000 时只有约 28%，9500 时近乎停滞。
    """
    total = 0
    for emotion, intensity in emotion_dict.items():
        coeff = AFFECTION_PER_INTENSITY.get(emotion, 0)
        if coeff != 0:
            total += int(coeff * intensity)

    # 软上限：只对正向增益做衰减，负向（扣好感）不受限制
    if total > 0 and current_affection > 0:
        saturation = current_affection / 10000  # 0~1
        decay = 1.0 - (saturation ** 1.5)       # 曲线：5000→0.65, 8000→0.28, 9500→0.08
        total = max(1, int(total * decay))

    return total


def make_emotion_context(emotions: dict, reasons: dict = None) -> str:
    """
    生成上下文式情绪状态字符串。
    如果有任何情绪 >= 1，返回 `[情绪状态: 标签1(强度)] [原因: xxx]`
    """
    if not emotions:
        return ""

    # 过滤有效情绪（>= 1 或 是基底情绪）
    valid = {k: v for k, v in emotions.items() if v >= 1}
    if not valid:
        return ""

    # 按强度排序，标注范围 0~100 让 LLM 理解数值含义
    sorted_emotions = sorted(valid.items(), key=lambda x: -x[1])
    parts = [f"{name}({int(round(val))}/100)" for name, val in sorted_emotions]
    emotion_str = " ".join(parts)

    # 取主导情绪的原因
    dominant = sorted_emotions[0][0]
    reason_str = ""
    if reasons and dominant in reasons:
        reason_str = reasons[dominant]

    if reason_str:
        result = f"[情绪状态: {emotion_str}] [原因: {reason_str}]"
    else:
        result = f"[情绪状态: {emotion_str}]"

    return result


def get_dominant_emotion(emotions: dict) -> tuple:
    """获取主导情绪 (emotion, intensity)"""
    if not emotions:
        return None, 0
    dominant = max(emotions.items(), key=lambda x: x[1])
    return dominant[0], dominant[1]


def detect_emotion_shift(reply_text: str, dominant_emotion: str, dominant_intensity: float) -> tuple:
    """
    检测 AI 回复中的情绪反转信号。
    返回 (target_emotion, transfer_amount) 或 None
    """
    if dominant_intensity < 60 or dominant_emotion not in SHIFT_SIGNALS:
        return None

    config = SHIFT_SIGNALS[dominant_emotion]
    for kw in config["keywords"]:
        if kw in reply_text:
            # 转移 50%
            transfer = int(dominant_intensity * 0.5)
            return (config["target"], transfer)
    return None


def resurrect_from_anchor(current_emotions: dict, anchor_emotion: str,
                          anchor_intensity: float) -> dict:
    """
    从情感锚点复活情绪。
    返回 {emotion: delta, ...}
    """
    delta = round(anchor_intensity * ANCHOR_RESURRECT_FACTOR, 1)
    updates = {anchor_emotion: delta}
    return updates
