"""
PersonaSim 轻量版 —— 4维状态引擎（energy / mood / social_need / satiety）

纯内存实现，无 DB 依赖，PersonaSim 插件不存在时可安全回退。
"""
import time
import random
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("qq-bot")

# ==================== 配置常量 ====================

# 每小时衰减/增长速率
ENERGY_DECAY_PER_HOUR = 3.0       # 活力每小时衰减
MOOD_DRIFT_PER_HOUR = 1.5         # 心情向基线(50)漂移
SOCIAL_NEED_GROWTH_PER_HOUR = 5.0 # 社交渴望每小时增长
SATIETY_DECAY_PER_HOUR = 7.0      # 饱腹感每小时衰减

MOOD_BASELINE = 50.0              # 心情基线

# 互动影响幅度
MOOD_POSITIVE_RANGE = (3, 8)      # 积极互动 → mood +3~8
MOOD_NORMAL_RANGE = (0, 3)        # 普通互动 → mood +0~3
MOOD_NEGATIVE_RANGE = (5, 12)     # 消极互动 → mood -5~12（取绝对值扣减）

SOCIAL_POSITIVE_RANGE = (5, 15)   # 积极互动 → social_need -5~15（满足感）
SOCIAL_NORMAL_RANGE = (2, 8)      # 普通互动 → social_need -2~8
SOCIAL_NEGATIVE_RANGE = (0, 5)    # 消极互动 → social_need +0~5（仍渴望连接）

ENERGY_INTERACTION_COST = (1, 4)  # 任何互动消耗 energy 1~4


@dataclass
class PersonaState:
    """4维角色状态，所有维度 0~100"""
    energy: float = 80.0
    mood: float = 60.0
    social_need: float = 40.0
    satiety: float = 65.0
    last_tick_at: float = 0.0


# ==================== 全局状态持有 ====================

_persona_states: dict[str, PersonaState] = {}


def get_state(scope_id: str) -> PersonaState:
    """获取或创建 scope 对应的角色状态。"""
    if scope_id not in _persona_states:
        _persona_states[scope_id] = PersonaState(last_tick_at=time.time())
    return _persona_states[scope_id]


def remove_state(scope_id: str) -> None:
    """清除状态（会话结束时调用）。"""
    _persona_states.pop(scope_id, None)


# ==================== 核心函数 ====================

def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def tick(state: PersonaState, now: float | None = None) -> PersonaState:
    """推进时间，应用自然衰减/增长。

    可重复调用，内部自动计算从上一次 tick 至今的时间差。
    """
    now = now or time.time()
    if state.last_tick_at <= 0:
        state.last_tick_at = now
        return state

    elapsed = (now - state.last_tick_at) / 3600.0
    if elapsed <= 0:
        return state

    logger.debug(f"[PersonaSim] tick scope elapsed={elapsed:.2f}h")

    # 活力衰减（随时间疲劳）
    state.energy = _clamp(state.energy - ENERGY_DECAY_PER_HOUR * elapsed)

    # 心情向基线漂移（情绪惯性）
    diff = state.mood - MOOD_BASELINE
    if abs(diff) > 0.5:
        drift = MOOD_DRIFT_PER_HOUR * elapsed * (1 if diff < 0 else -1)
        state.mood = _clamp(state.mood + drift)

    # 社交渴望增长（独孤感积累）
    state.social_need = _clamp(state.social_need + SOCIAL_NEED_GROWTH_PER_HOUR * elapsed)

    # 饱腹感衰减（逐渐变饿）
    state.satiety = _clamp(state.satiety - SATIETY_DECAY_PER_HOUR * elapsed)

    state.last_tick_at = now
    return state


def apply_interaction(
    state: PersonaState,
    quality: str = "normal",
) -> PersonaState:
    """根据互动质量更新 4维状态。

    Args:
        state: 当前状态
        quality: "positive" | "normal" | "negative"
    """
    if quality == "positive":
        mood_delta = random.uniform(*MOOD_POSITIVE_RANGE)
        social_delta = -random.uniform(*SOCIAL_POSITIVE_RANGE)
    elif quality == "negative":
        mood_delta = -random.uniform(*MOOD_NEGATIVE_RANGE)
        social_delta = random.uniform(*SOCIAL_NEGATIVE_RANGE)
    else:  # normal
        mood_delta = random.uniform(*MOOD_NORMAL_RANGE)
        social_delta = -random.uniform(*SOCIAL_NORMAL_RANGE)

    energy_cost = random.uniform(*ENERGY_INTERACTION_COST)

    state.mood = _clamp(state.mood + mood_delta)
    state.social_need = _clamp(state.social_need + social_delta)
    state.energy = _clamp(state.energy - energy_cost)

    logger.debug(
        f"[PersonaSim] apply_interaction quality={quality} "
        f"mood={mood_delta:+.1f} social={social_delta:+.1f} energy={-energy_cost:+.1f}"
    )
    return state


# ==================== 外部插件集成 ====================

def try_apply_external_interaction(
    scope_id: str,
    quality: str = "normal",
    now: float | None = None,
) -> PersonaState | None:
    """尝试通过外部 PersonaSim 插件 apply_interaction。

    外部插件需提供 person_sim_engine 实例，挂载在全局可访问的位置。
    本函数先检查插件是否存在，存在则委托给插件，否则用本地引擎。
    """
    engine = _get_external_engine()
    if engine is not None:
        try:
            import asyncio
            coro = engine.apply_interaction(scope_id, quality=quality, mode="passive", now=now)
            # 如果在事件循环中，安全地创建任务
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception as e:
            logger.warning(f"[PersonaSim] 外部引擎调用失败, 回退本地: {e}")
            return _apply_local(scope_id, quality, now)
        return None  # 外部引擎异步执行，不返回同步结果
    return _apply_local(scope_id, quality, now)


def _apply_local(scope_id: str, quality: str, now: float | None) -> PersonaState:
    state = get_state(scope_id)
    tick(state, now)
    apply_interaction(state, quality)
    return state


def _get_external_engine():
    """获取外部 PersonaSim 引擎实例（如 astrbot 插件注册的）。"""
    try:
        import sys
        if "persona_sim_engine" in sys.modules:
            # 尝试从已知位置获取
            from astrbot_plugin_self_evolution.engine.persona_sim_engine import PersonaSimEngine
            # 查找已注册的实例
            for obj in sys.modules.values():
                if isinstance(obj, PersonaSimEngine):
                    return obj
                for attr_name in dir(obj):
                    try:
                        attr = getattr(obj, attr_name)
                        if isinstance(attr, PersonaSimEngine):
                            return attr
                    except Exception:
                        continue
    except (ImportError, AttributeError):
        pass
    return None
