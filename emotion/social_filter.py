"""
社交过滤器 + 认知资源分配权重

傲娇人格下，真实情绪和表达情绪之间存在"社交过滤器"，
每次抑制消耗认知资源，资源随现实时间恢复。
好感度决定资源容器大小，实现"低好感敷衍/高好感深聊"行为。

使用方式：
    state = CognitiveResourceManager.get(session_id)
    state.update_capacity(affection)
    filtered, _ = apply_social_filter(emotions, reasons, user_emotion, state)
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
        "name": "不熟_醋意掩饰为冷漠",
        "condition": lambda e, a: e.get("醋意", 0) >= 7 and 0 <= a < 3000,
        "skip_suppression": None,
        "transform": lambda expressed, orig: {
            **expressed,
            "醋意": round(orig.get("醋意", 0) * 0.3, 1),
            "生气": expressed.get("生气", 0) + round(orig.get("醋意", 0) * 0.4, 1),
        },
        "cost": 0.08,
    },
    {
        "name": "信任_直接表达脆弱",
        "condition": lambda e, a: e.get("委屈", 0) >= 6 and a > 7000,
        "skip_suppression": {"委屈"},
        "transform": None,
        "cost": -0.03,
    },
]


def _reverse_affection(max_capacity: float) -> float:
    """从 max_capacity 反推好感度（与 update_capacity 互逆）"""
    if RESOURCE_CAPACITY_MAX <= RESOURCE_CAPACITY_MIN:
        return 0.0
    ratio = (max_capacity - RESOURCE_CAPACITY_MIN) / (RESOURCE_CAPACITY_MAX - RESOURCE_CAPACITY_MIN)
    return ratio * 10000


class CognitiveResourceState:
    """认知资源分配状态"""
    __slots__ = (
        'session_id', 'current', 'max_capacity',
        'last_update', 'suppression_count', 'burst_mode',
    )

    def __init__(
        self, session_id: str, current: float = 60.0,
        max_capacity: float = 60.0, last_update: float = 0.0,
        suppression_count: int = 0, burst_mode: bool = False,
    ):
        self.session_id = session_id
        self.current = current
        self.max_capacity = max_capacity
        self.last_update = last_update
        self.suppression_count = suppression_count
        self.burst_mode = burst_mode

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
        self.max_capacity = (
            RESOURCE_CAPACITY_MIN
            + norm * (RESOURCE_CAPACITY_MAX - RESOURCE_CAPACITY_MIN)
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


def apply_social_filter(
    emotions: dict,
    reasons: dict,
    user_emotions: list[str],
    state: CognitiveResourceState,
) -> tuple[dict, dict]:
    """
    对内部情绪施加社交过滤器。

    返回 (filtered_emotions, filtered_reasons)

    傲娇过滤规则:
    1. 害羞 >= 3 -> 抑制 50%，补偿撒娇 +1
    2. 开心 >= 4 且被夸(用户情绪含"害羞") -> 降级(开心-2 -> 害羞+1)
    3. 醋意 >= 3 -> 抑制 40%
    4. 委屈 >= 6 -> 抑制 50%，假装没事
    """
    state.recover()
    ratio = state.ratio

    expressed = dict(emotions)
    exp_reasons = dict(reasons)
    total_cost = 0.0

    # ---- 爆发检测与处理 ----
    if state.burst_mode and ratio > BURST_EXIT_THRESHOLD:
        state.burst_mode = False
        print(f"[社交过滤器] 资源恢复({ratio:.0%})，退出爆发模式")

    in_burst = state.burst_mode or (
        ratio <= BURST_THRESHOLD and state.suppression_count > 0
    )

    if in_burst:
        if not state.burst_mode:
            print(f"[社交过滤器] 资源耗尽({ratio:.0%})，进入爆发模式")
            state.burst_mode = True
        # 爆发模式：不抑制，增强表达 1.5 倍
        for k in list(expressed.keys()):
            expressed[k] = min(10, expressed[k] * 1.5)
        state.suppression_count += 1
        print(f"[社交过滤器] 爆发: {dict(emotions)} -> {expressed}")
        _cleanup(expressed, exp_reasons)
        return expressed, exp_reasons

    # ---- 伪装规则预检查：跳过抑制 ----
    affection = _reverse_affection(state.max_capacity)
    skip_suppression_labels: set[str] = set()
    for rule in DISGUISE_RULES:
        if rule.get("skip_suppression") and rule["condition"](emotions, affection):
            skip_suppression_labels.update(rule["skip_suppression"])
            print(
                f"[社交过滤器] 伪装规则: {rule['name']}"
                f" → 跳过 {rule['skip_suppression']} 的抑制"
            )

    # ---- 正常抑制（检查跳过标签） ----

    # 规则 1: 害羞 >= 3 抑制 50%，补撒娇
    if "害羞" not in skip_suppression_labels and expressed.get("害羞", 0) >= 3:
        v = expressed["害羞"]
        expressed["害羞"] = round(v * 0.5, 1)
        expressed["撒娇"] = expressed.get("撒娇", 0) + 1
        total_cost += 0.05
        print(f"[社交过滤器] 害羞 {v:.1f}->{expressed['害羞']:.1f}，撒娇+1")

    # 规则 2: 开心 >= 4 且被夸(用户情绪含害羞) -> 傲娇否认
    if "开心" not in skip_suppression_labels and expressed.get("开心", 0) >= 4 and "害羞" in user_emotions:
        v = expressed["开心"]
        expressed["开心"] = max(1, v - 2)
        expressed["害羞"] = expressed.get("害羞", 0) + 1
        total_cost += 0.08
        print(f"[社交过滤器] 开心 {v:.1f}->{expressed['开心']:.1f}，害羞+1")

    # 规则 3: 醋意 >= 3 抑制 40%
    if "醋意" not in skip_suppression_labels and expressed.get("醋意", 0) >= 3:
        v = expressed["醋意"]
        expressed["醋意"] = round(v * 0.6, 1)
        total_cost += 0.06
        print(f"[社交过滤器] 醋意 {v:.1f}->{expressed['醋意']:.1f}")

    # 规则 4: 委屈 >= 6 抑制 50%
    if "委屈" not in skip_suppression_labels and expressed.get("委屈", 0) >= 6:
        v = expressed["委屈"]
        expressed["委屈"] = round(v * 0.5, 1)
        total_cost += 0.04
        print(f"[社交过滤器] 委屈 {v:.1f}->{expressed['委屈']:.1f}")

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
    # 确保所有情绪不超过上限 10
    for k in list(expressed.keys()):
        expressed[k] = min(10, expressed[k])
    return expressed, exp_reasons


def _cleanup(emotions: dict, reasons: dict):
    """清理低于阈值(1)的情绪"""
    for k in list(emotions.keys()):
        if emotions[k] < 1:
            del emotions[k]
            reasons.pop(k, None)
