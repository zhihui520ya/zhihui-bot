"""
社交过滤器 + 认知资源分配权重

傲娇人格下，真实情绪和表达情绪之间存在"社交过滤器"，
每次抑制消耗认知资源，资源随现实时间恢复。
好感度决定资源容器大小，实现"低好感敷衍/高好感深聊"行为。

使用方式：
    state = CognitiveResourceManager.get(session_id)
    state.update_capacity(affection)
    filtered, _, narrative = apply_social_filter(emotions, reasons, user_emotion, state, affection)
    CognitiveResourceManager.save(session_id)
"""

import time
import json
import os
from config import (
    COGNITIVE_CAPACITY_MIN,
    COGNITIVE_CAPACITY_MAX,
    COGNITIVE_RECOVERY_RATE,
    COGNITIVE_BURST_THRESHOLD,
    COGNITIVE_BURST_EXIT_THRESHOLD,
    DATA_DIR,
)

# ========== 配置 ==========

# 资源容量映射：好感度 -10~10 → 最大资源量
# 低好感(=-10) → 容器小(=20) → 灌满块 → 快速回满但一次就用完 → 敷衍
# 高好感(=10) → 容器大(=100) → 灌满慢 → 缓慢回满但经得起消耗 → 深聊
RESOURCE_CAPACITY_MIN = COGNITIVE_CAPACITY_MIN
RESOURCE_CAPACITY_MAX = COGNITIVE_CAPACITY_MAX

# 恢复速率
RECOVERY_RATE_PER_SEC = COGNITIVE_RECOVERY_RATE

# 爆发阈值
BURST_THRESHOLD = COGNITIVE_BURST_THRESHOLD
BURST_EXIT_THRESHOLD = COGNITIVE_BURST_EXIT_THRESHOLD
BURST_COOLDOWN = 36 * 3600  # 36小时爆发冷却

STORAGE_DIR = os.path.join(DATA_DIR, "memories", "cognitive_resources")

# ========== 伪装策略规则 ==========
# 在傲娇抑制之前（检查跳过）、之后（变换）执行
# condition: (original_emotions, affection) → bool
#   original_emotions: 原始的未经任何修改的情绪 dict（第 1 遍 check 用）
# skip_suppression: 不为 None 时，被抑制前检查，跳过集合内标签的抑制规则
# transform: 不为 None 时，抑制后执行变换 → (expressed, orig_emotions) → 新 expressed
# cost: 额外认知资源消耗（负值 = 节省，因为不需要伪装）

DISGUISE_RULES = [
    {
        "name": "不熟_恐惧掩饰为愤怒",
        "condition": lambda e, a: e.get("恐惧", 0) >= 70 and 0 <= a < 3000,
        "skip_suppression": None,
        "transform": lambda expressed, orig: {
            **expressed,
            "恐惧": round(orig.get("恐惧", 0) * 0.3, 1),
            "愤怒": expressed.get("愤怒", 0) + round(orig.get("恐惧", 0) * 0.4, 1),
        },
        "cost": 0.4,
    },
    {
        "name": "信任_直接表达脆弱",
        "condition": lambda e, a: e.get("悲伤", 0) >= 60 and a > 7000,
        "skip_suppression": {"悲伤"},
        "transform": None,
        "cost": -0.15,
    },
]


class CognitiveResourceState:
    """认知资源分配状态"""
    __slots__ = (
        'session_id', 'current', 'max_capacity',
        'last_update', 'suppression_count', 'burst_mode',
        'last_burst_at',
    )

    def __init__(
        self, session_id: str, current: float = 60.0,
        max_capacity: float = 60.0, last_update: float = 0.0,
        suppression_count: int = 0, burst_mode: bool = False,
        last_burst_at: float = 0.0,
    ):
        self.session_id = session_id
        self.current = current
        self.max_capacity = max_capacity
        self.last_update = last_update
        self.suppression_count = suppression_count
        self.burst_mode = burst_mode
        self.last_burst_at = last_burst_at

    def recover(self):
        """随时间恢复认知资源"""
        now = time.time()
        if self.last_update > 0:
            elapsed = now - self.last_update
            if elapsed > 0:
                recovered = elapsed * RECOVERY_RATE_PER_SEC
                self.current = min(self.max_capacity, self.current + recovered)
        self.last_update = now

    def update_capacity(self, affection: float):
        """根据好感度更新最大容量"""
        norm = affection / 10000  # 0~10000 → 0~1
        # 好感度低→容量大（更克制、更会装）
        # 好感度高→容量小（容易装累→爆发→真情流露）
        self.max_capacity = (
            RESOURCE_CAPACITY_MAX
            - norm * (RESOURCE_CAPACITY_MAX - RESOURCE_CAPACITY_MIN)
        )
        self.current = min(self.current, self.max_capacity)

    @property
    def ratio(self) -> float:
        return self.current / self.max_capacity if self.max_capacity > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "current": round(self.current, 2),
            "max_capacity": round(self.max_capacity, 2),
            "last_update": self.last_update,
            "suppression_count": self.suppression_count,
            "burst_mode": self.burst_mode,
            "last_burst_at": self.last_burst_at,
        }

    @classmethod
    def from_dict(cls, session_id: str, d: dict) -> "CognitiveResourceState":
        return cls(
            session_id=session_id,
            current=d["current"],
            max_capacity=d["max_capacity"],
            last_update=d["last_update"],
            suppression_count=d.get("suppression_count", 0),
            burst_mode=d.get("burst_mode", False),
            last_burst_at=d.get("last_burst_at", 0.0),
        )


class CognitiveResourceManager:
    """管理所有会话的认知资源状态"""
    _instances: dict[str, CognitiveResourceState] = {}

    @classmethod
    def get(cls, session_id: str) -> CognitiveResourceState:
        if session_id not in cls._instances:
            state = cls._load(session_id)
            if state is None:
                state = CognitiveResourceState(session_id=session_id)
            cls._instances[session_id] = state
        return cls._instances[session_id]

    @classmethod
    def save(cls, session_id: str):
        state = cls._instances.get(session_id)
        if state:
            cls._save(session_id, state)

    @classmethod
    def save_all(cls):
        for sid in list(cls._instances.keys()):
            cls.save(sid)

    @classmethod
    def _load(cls, session_id: str) -> CognitiveResourceState | None:
        fpath = os.path.join(STORAGE_DIR, f"{session_id}.json")
        try:
            with open(fpath) as f:
                return CognitiveResourceState.from_dict(session_id, json.load(f))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    @classmethod
    def _save(cls, session_id: str, state: CognitiveResourceState):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        fpath = os.path.join(STORAGE_DIR, f"{session_id}.json")
        with open(fpath, "w") as f:
            json.dump(state.to_dict(), f)


def _affection_scale(affection: float, low: float, high: float) -> float:
    """好感度 0~10000 → 低值~高值 线性插值。好感越低越抑制（低值），好感越高越真实（高值）"""
    norm = min(1.0, affection / 10000)
    return low + (high - low) * norm


# ========== 傲娇抑制 → 行为描述词汇 ==========

_SUPPRESSION_NARRATIVES = {
    "快乐": {
        "inner": {
            (0, 20): "心情一般",
            (20, 40): "心情还行",
            (40, 60): "心里有点高兴",
            (60, 80): "其实挺开心的",
            (80, float("inf")): "内心超开心的",
        },
        "cover": [
            "但故意装作不在意的样子",
            "但板着脸说'还行吧'",
            "但嘴上说'才没有呢'",
            "但假装不是很在意",
        ],
    },
    "悲伤": {
        "inner": {
            (0, 20): "没什么",
            (20, 40): "稍微有点低落",
            (40, 60): "有点难受",
            (60, 80): "其实挺难过的",
            (80, float("inf")): "心里很难受",
        },
        "cover": [
            "但假装没事",
            "但强颜欢笑",
            "但笑着说'我没事'",
        ],
    },
    "恐惧": {
        "inner": {
            (0, 20): "没什么好怕的",
            (20, 40): "稍微有点在意",
            (40, 60): "其实有点不安",
            (60, 80): "其实挺害怕的",
            (80, float("inf")): "内心很害怕",
        },
        "cover": [
            "但强装镇定",
            "但故作坚强",
            "但嘴硬说'谁怕了'",
        ],
    },
    "愤怒": {
        "inner": {
            (0, 20): "没什么",
            (20, 40): "稍微有点不爽",
            (40, 60): "有点烦躁",
            (60, 80): "其实有点生气",
            (80, float("inf")): "其实非常生气",
        },
        "cover": [
            "但努力保持冷静",
            "但压住了火气",
            "但假装没在生气",
        ],
    },
    "惊讶": {
        "inner": {
            (0, 20): "没什么特别的",
            (20, 40): "有点意外",
            (40, 60): "确实没想到",
            (60, 80): "确实被惊到了",
            (80, float("inf")): "完全出乎意料",
        },
        "cover": [
            "但装作没什么大不了的",
            "但表面上一脸平静",
            "但故作淡定地说'哦'",
        ],
    },
    "厌恶": {
        "inner": {
            (0, 20): "没什么",
            (20, 40): "有点不舒服",
            (40, 60): "不太喜欢",
            (60, 80): "其实挺反感的",
            (80, float("inf")): "非常讨厌",
        },
        "cover": [
            "但忍住了没说",
            "但装作无所谓的样子",
            "但压下了那股情绪",
        ],
    },
}


def _pick_narrative(emotion: str, raw_val: float) -> str:
    """根据原始情绪强度选取内心描述"""
    ranges = _SUPPRESSION_NARRATIVES.get(emotion, {}).get("inner", {})
    for (lo, hi), text in sorted(ranges.items()):
        if lo <= raw_val < hi:
            return text
    return ""


def _pick_cover(emotion: str) -> str:
    """随机选取一种傲娇行为表现"""
    covers = _SUPPRESSION_NARRATIVES.get(emotion, {}).get("cover", [])
    import random
    return random.choice(covers) if covers else "但不想表现出来"


def apply_social_filter(
    emotions: dict,
    reasons: dict,
    user_emotions: list[str],
    state: CognitiveResourceState,
    affection: float = 0.0,
) -> tuple[dict, dict, str]:
    """
    对内部情绪施加社交过滤器。

    返回 (expressed_emotions, expressed_reasons, narrative)
      - expressed_emotions: 抑制后的情绪值（供其他系统使用）
      - expressed_reasons: 对应原因
      - narrative: 本地生成的傲娇行为描述文本，供 LLM 使用

    傲娇过滤规则（所有抑制比例按好感度动态计算）:
    1. 快乐 >= 40 且被夸 → 快乐×0.4~0.8，惊讶+0.1~0.3
    2. 悲伤 >= 60 → 悲伤×0.3~0.7，愤怒+0.1~0.3（藏悲→烦躁）
    3. 恐惧 >= 40 → 恐惧×0.4~0.7，愤怒+0.1~0.3（虚张声势）
    4. 愤怒 >= 60 → 愤怒×0.5~0.8，恐惧+0.0~0.2（压下火气后不安）
    5. 惊讶 >= 40 → 惊讶×0.5~0.8，快乐+0.1~0.2（故作淡定）
    6. 厌恶 >= 40 → 厌恶×0.4~0.7，愤怒+0.1~0.3（忍着不说）
    """
    state.recover()
    ratio = state.ratio

    expressed = dict(emotions)
    exp_reasons = dict(reasons)
    total_cost = 0.0
    narrative_parts = []  # 收集行为描述片段

    # ---- 爆发检测与处理 ----
    if state.burst_mode and ratio > BURST_EXIT_THRESHOLD:
        state.burst_mode = False
        print(f"[社交过滤器] 资源恢复({ratio:.0%})，退出爆发模式")

    _can_burst = (time.time() - state.last_burst_at) >= BURST_COOLDOWN
    in_burst = state.burst_mode or (
        ratio <= BURST_THRESHOLD and state.suppression_count > 0 and _can_burst
    )

    if in_burst:
        if not state.burst_mode:
            print(f"[社交过滤器] 资源耗尽({ratio:.0%})，进入爆发模式")
            state.burst_mode = True
        state.last_burst_at = time.time()
        # 爆发模式：不抑制，正负情绪区分倍数
        _positive_emotions = {"快乐", "惊讶"}
        for k in list(expressed.keys()):
            if k in _positive_emotions:
                expressed[k] = min(100, expressed[k] * 1.3)
            else:
                expressed[k] = min(100, expressed[k] * 1.5)
        state.suppression_count += 1
        print(f"[社交过滤器] 爆发: {dict(emotions)} -> {expressed}")
        _cleanup(expressed, exp_reasons)
        narrative_parts.append("情绪压不住了，全部涌了出来")
        narrative = "，".join(narrative_parts) if narrative_parts else ""
        print(f"[社交过滤器] 爆发描述: {narrative}")
        return expressed, exp_reasons, narrative

    # ---- 伪装规则预检查：跳过抑制 ----
    skip_suppression_labels: set[str] = set()
    for rule in DISGUISE_RULES:
        if rule.get("skip_suppression") and rule["condition"](emotions, affection):
            skip_suppression_labels.update(rule["skip_suppression"])
            print(
                f"[社交过滤器] 伪装规则: {rule['name']}"
                f" → 跳过 {rule['skip_suppression']} 的抑制"
            )

    # ---- 正常抑制（检查跳过标签），动态比例 ----

    # 规则 1: 快乐 >= 40 且被夸 -> 傲娇否认
    if "快乐" not in skip_suppression_labels and expressed.get("快乐", 0) >= 40 and "快乐" in user_emotions:
        raw = expressed["快乐"]
        scale = _affection_scale(affection, 0.4, 0.8)
        expressed["快乐"] = round(raw * scale, 1)
        comp = round(_affection_scale(affection, 0.1, 0.3), 1)
        expressed["惊讶"] = expressed.get("惊讶", 0) + comp
        total_cost += 0.4
        inner_text = _pick_narrative("快乐", raw)
        cover_text = _pick_cover("快乐")
        narrative_parts.append(f"{inner_text}，{cover_text}")
        print(f"[社交过滤器] 快乐 {raw:.1f}×{scale:.2f}->{expressed['快乐']:.1f}，惊讶+{comp}")

    # 规则 2: 悲伤 >= 60 -> 假装没事
    if "悲伤" not in skip_suppression_labels and expressed.get("悲伤", 0) >= 60:
        raw = expressed["悲伤"]
        scale = _affection_scale(affection, 0.3, 0.7)
        expressed["悲伤"] = round(raw * scale, 1)
        comp = round(_affection_scale(affection, 0.1, 0.3), 1)
        expressed["愤怒"] = expressed.get("愤怒", 0) + comp
        total_cost += 0.2
        inner_text = _pick_narrative("悲伤", raw)
        cover_text = _pick_cover("悲伤")
        narrative_parts.append(f"{inner_text}，{cover_text}")
        print(f"[社交过滤器] 悲伤 {raw:.1f}×{scale:.2f}->{expressed['悲伤']:.1f}，愤怒+{comp}")

    # 规则 3: 恐惧 >= 40 -> 虚张声势
    if "恐惧" not in skip_suppression_labels and expressed.get("恐惧", 0) >= 40:
        raw = expressed["恐惧"]
        scale = _affection_scale(affection, 0.4, 0.7)
        expressed["恐惧"] = round(raw * scale, 1)
        comp = round(_affection_scale(affection, 0.1, 0.3), 1)
        expressed["愤怒"] = expressed.get("愤怒", 0) + comp
        total_cost += 0.3
        inner_text = _pick_narrative("恐惧", raw)
        cover_text = _pick_cover("恐惧")
        narrative_parts.append(f"{inner_text}，{cover_text}")
        print(f"[社交过滤器] 恐惧 {raw:.1f}×{scale:.2f}->{expressed['恐惧']:.1f}，愤怒+{comp}")

    # 规则 4: 愤怒 >= 60 -> 压住火气
    if "愤怒" not in skip_suppression_labels and expressed.get("愤怒", 0) >= 60:
        raw = expressed["愤怒"]
        scale = _affection_scale(affection, 0.5, 0.8)
        expressed["愤怒"] = round(raw * scale, 1)
        comp = round(_affection_scale(affection, 0.0, 0.2), 1)
        expressed["恐惧"] = expressed.get("恐惧", 0) + comp
        total_cost += 0.2
        inner_text = _pick_narrative("愤怒", raw)
        cover_text = _pick_cover("愤怒")
        narrative_parts.append(f"{inner_text}，{cover_text}")
        print(f"[社交过滤器] 愤怒 {raw:.1f}×{scale:.2f}->{expressed['愤怒']:.1f}，恐惧+{comp}")

    # 规则 5: 惊讶 >= 40 -> 故作淡定
    if "惊讶" not in skip_suppression_labels and expressed.get("惊讶", 0) >= 40:
        raw = expressed["惊讶"]
        scale = _affection_scale(affection, 0.5, 0.8)
        expressed["惊讶"] = round(raw * scale, 1)
        comp = round(_affection_scale(affection, 0.1, 0.2), 1)
        expressed["快乐"] = expressed.get("快乐", 0) + comp
        total_cost += 0.2
        inner_text = _pick_narrative("惊讶", raw)
        cover_text = _pick_cover("惊讶")
        narrative_parts.append(f"{inner_text}，{cover_text}")
        print(f"[社交过滤器] 惊讶 {raw:.1f}×{scale:.2f}->{expressed['惊讶']:.1f}，快乐+{comp}")

    # 规则 6: 厌恶 >= 40 -> 忍着不说
    if "厌恶" not in skip_suppression_labels and expressed.get("厌恶", 0) >= 40:
        raw = expressed["厌恶"]
        scale = _affection_scale(affection, 0.4, 0.7)
        expressed["厌恶"] = round(raw * scale, 1)
        comp = round(_affection_scale(affection, 0.1, 0.3), 1)
        expressed["愤怒"] = expressed.get("愤怒", 0) + comp
        total_cost += 0.3
        inner_text = _pick_narrative("厌恶", raw)
        cover_text = _pick_cover("厌恶")
        narrative_parts.append(f"{inner_text}，{cover_text}")
        print(f"[社交过滤器] 厌恶 {raw:.1f}×{scale:.2f}->{expressed['厌恶']:.1f}，愤怒+{comp}")

    # ---- 伪装规则后处理：变换 ----
    for rule in DISGUISE_RULES:
        if rule.get("transform") and rule["condition"](emotions, affection):
            old = dict(expressed)
            expressed = rule["transform"](expressed, emotions)
            total_cost += rule.get("cost", 0)
            print(f"[社交过滤器] 伪装规则: {rule['name']}: {old} -> {expressed}")

    # ---- 扣除资源 ----
    if total_cost > 0:
        state.current = max(0.0, state.current - total_cost)
        state.suppression_count += 1
    else:
        state.suppression_count = 0

    print(
        f"[社交过滤器] 资源: {state.current:.1f}/{state.max_capacity:.0f}({ratio:.0%}), "
        f"消耗: {total_cost:.3f}"
    )

    _cleanup(expressed, exp_reasons)
    for k in list(expressed.keys()):
        expressed[k] = min(100, expressed[k])

    # 组装最终描述
    narrative = "，".join(narrative_parts) if narrative_parts else ""
    return expressed, exp_reasons, narrative


def _cleanup(emotions: dict, reasons: dict):
    """清理低于阈值(1)的情绪"""
    for k in list(emotions.keys()):
        if emotions[k] < 1:
            del emotions[k]
            reasons.pop(k, None)
