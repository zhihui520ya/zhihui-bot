"""
PersonaSim ↔ 知慧1情绪系统 桥接层

职责：
  1. PersonaSim 4维 → 6种基础情绪偏置（persona_to_emotion_bias）
  2. 知慧1情绪变化 → PersonaSim mood 反馈（emotion_to_persona_feedback）
  3. PersonaSim 上下文注入 LLM（get_persona_context）
  4. Social_need → 回复概率修正（get_reply_probability_modifier）
"""
import logging
from .persona_sim import PersonaState, tick, apply_interaction, get_state

logger = logging.getLogger("qq-bot")

# ==================== PersonaSim → 情绪偏置 ====================

# 阈值
MOOD_HIGH = 70       # 心情高阈值
MOOD_LOW = 30        # 心情低阈值
SOCIAL_HIGH = 70     # 社交渴望高阈值
ENERGY_HIGH = 70     # 活力高阈值
ENERGY_LOW = 30      # 活力低阈值
SATIETY_HIGH = 70    # 饱腹高阈值
SATIETY_LOW = 30     # 饱腹低阈值


def persona_to_emotion_bias(state: PersonaState) -> dict[str, float]:
    """根据 PersonaSim 4维状态计算情绪偏置量。

    Returns:
        {emotion_label: delta}，正值提升该情绪强度，负值抑制。
        调用方将这些 delta 叠加到现有的 7 维情绪向量上。
    """
    bias: dict[str, float] = {}

    # ---- mood 驱动 ----
    if state.mood >= MOOD_HIGH:
        # 心情好 → 快乐增强，负面情绪减弱
        strength = (state.mood - MOOD_HIGH) / (100 - MOOD_HIGH)  # 0~1
        bias["快乐"] = round(strength * 3.0, 1)  # 0~3
        bias["愤怒"] = -round(strength * 1.0, 1)  # 0~-1
        bias["悲伤"] = -round(strength * 0.5, 1)  # 0~-0.5
    elif state.mood <= MOOD_LOW:
        # 心情差 → 负面情绪增强，快乐减弱
        strength = (MOOD_LOW - state.mood) / MOOD_LOW  # 0~1
        bias["悲伤"] = round(strength * 2.0, 1)  # 0~2
        bias["愤怒"] = round(strength * 1.5, 1)  # 0~1.5
        bias["快乐"] = -round(strength * 1.0, 1)  # 0~-1

    # ---- social_need 驱动 ----
    if state.social_need >= SOCIAL_HIGH:
        # 渴望社交 → 恐惧/快乐（既期待又不安）
        strength = (state.social_need - SOCIAL_HIGH) / (100 - SOCIAL_HIGH)
        bias["恐惧"] = bias.get("恐惧", 0) + round(strength * 2.0, 1)  # 0~2
        bias["快乐"] = bias.get("快乐", 0) + round(strength * 1.5, 1)  # 0~1.5

    # ---- energy 驱动 ----
    if state.energy >= ENERGY_HIGH:
        # 活力充沛 → 快乐/惊讶
        strength = (state.energy - ENERGY_HIGH) / (100 - ENERGY_HIGH)
        bias["快乐"] = bias.get("快乐", 0) + round(strength * 1.5, 1)
        bias["惊讶"] = bias.get("惊讶", 0) + round(strength * 1.0, 1)
    elif state.energy <= ENERGY_LOW:
        # 疲劳 → 悲伤倾向
        strength = (ENERGY_LOW - state.energy) / ENERGY_LOW
        bias["悲伤"] = bias.get("悲伤", 0) + round(strength * 1.0, 1)

    # ---- satiety 驱动 ----
    if state.satiety >= SATIETY_HIGH:
        # 饱足 → 满足稳定，微微快乐
        bias["快乐"] = bias.get("快乐", 0) + 0.5
    elif state.satiety <= SATIETY_LOW:
        # 饥饿 → 容易烦躁
        strength = (SATIETY_LOW - state.satiety) / SATIETY_LOW
        bias["愤怒"] = bias.get("愤怒", 0) + round(strength * 1.0, 1)

    # 过滤掉绝对值 < 0.5 的微弱偏置
    bias = {k: v for k, v in bias.items() if abs(v) >= 0.5}

    if bias:
        logger.debug(f"[PersonaBridge] persona→emotion bias: {bias}")
    return bias


# ==================== 情绪 → PersonaSim 反馈 ====================


def emotion_to_persona_feedback(emotion_dict: dict[str, float]) -> dict[str, float]:
    """根据知慧1情绪变化量计算 PersonaSim 状态反馈。

    Args:
        emotion_dict: 当前情绪向量 {label: intensity}

    Returns:
        {"mood": delta, "energy": delta, ...} 正值增加，负值减少。
    """
    deltas: dict[str, float] = {}

    # 正向情绪 → mood 提升
    happy = emotion_dict.get("快乐", 0)
    surprised = emotion_dict.get("惊讶", 0)

    positive_total = happy * 0.8 + surprised * 0.3
    if positive_total >= 2.0:
        deltas["mood"] = round(min(positive_total * 0.5, 8.0), 1)

    # 负向情绪 → mood 降低
    angry = emotion_dict.get("愤怒", 0)
    sad = emotion_dict.get("悲伤", 0)
    fearful = emotion_dict.get("恐惧", 0)

    negative_total = angry * 1.5 + sad * 1.0 + fearful * 0.8
    if negative_total >= 2.0:
        deltas["mood"] = deltas.get("mood", 0) - round(min(negative_total * 0.6, 10.0), 1)

    # 快乐 → social_need 降低（社交满足）
    if happy >= 3:
        deltas["social_need"] = -round(happy * 1.5, 1)
    # 恐惧 → social_need 提高（寻求安慰）
    if fearful >= 3:
        deltas["social_need"] = deltas.get("social_need", 0) + round(fearful * 1.0, 1)

    # 愤怒/悲伤 → energy 额外消耗
    if angry >= 5 or sad >= 5:
        deltas["energy"] = -round(max(angry, sad) * 0.8, 1)

    if deltas:
        logger.debug(f"[PersonaBridge] emotion→persona feedback: {deltas}")
    return deltas


# ==================== 上下文生成 ====================


def get_persona_context(state: PersonaState) -> str:
    """生成 PersonaSim 状态的自然语言描述，注入 LLM 上下文。"""
    parts = []
    # 活力
    if state.energy >= 80:
        parts.append("精力充沛")
    elif state.energy >= 60:
        parts.append("精神还不错")
    elif state.energy >= 40:
        parts.append("有点疲惫")
    elif state.energy >= 20:
        parts.append("很累")
    else:
        parts.append("精疲力竭")

    # 社交渴望
    if state.social_need >= 80:
        parts.append("渴望和人说话")
    elif state.social_need >= 60:
        parts.append("有点想找人聊天")
    elif state.social_need >= 40:
        parts.append("还好")
    elif state.social_need < 20:
        parts.append("不太想理人")

    # 饱腹感
    if state.satiety >= 80:
        parts.append("吃饱了")
    elif state.satiety >= 50:
        parts.append("不饿")
    elif state.satiety >= 30:
        parts.append("有点饿")
    else:
        parts.append("好饿")

    result = " / ".join(parts)
    return f"[Persona: {result}]"


# ==================== 回复概率修正 ====================


def get_reply_probability_modifier(state: PersonaState) -> float:
    """根据 PersonaSim 状态计算回复概率修正因子。

    Returns:
        乘法因子，1.0 = 无变化，>1 = 增加回复倾向，<1 = 减少回复倾向。
    """
    modifier = 1.0

    # social_need: 孤独感越强越想回复
    if state.social_need > 70:
        boost = (state.social_need - 70) / 30 * 0.4  # 0~0.4
        modifier += boost
    elif state.social_need < 30:
        modifier -= 0.15

    # energy: 太累了不想说话
    if state.energy < 30:
        modifier -= 0.15
    elif state.energy > 80:
        modifier += 0.1

    # mood: 心情太差不想说话
    if state.mood < 25:
        modifier -= 0.15

    modifier = max(0.4, min(1.6, modifier))
    return modifier


# ==================== 一站式入口：处理完整一轮互动 ====================


def process_interaction(
    scope_id: str,
    emotion_dict: dict[str, float],
    interaction_quality: str = "normal",
) -> dict[str, float]:
    """处理一次完整的 PersonaSim 互动周期。

    1. tick 推进时间
    2. apply_interaction 更新状态
    3. 计算情绪偏置
    4. 计算情绪反馈
    5. 返回叠加后的 emotion_dict

    Args:
        scope_id: 会话 ID
        emotion_dict: 当前 6 维情绪向量（会被偏置修改）
        interaction_quality: "positive" | "normal" | "negative"

    Returns:
        叠加了 persona 偏置后的新 emotion_dict（浅拷贝 + 修改）
    """
    import copy
    state = get_state(scope_id)
    tick(state)

    # 互动影响
    apply_interaction(state, quality=interaction_quality)

    # 4维 → 情绪偏置
    bias = persona_to_emotion_bias(state)
    result = copy.copy(emotion_dict)  # 浅拷贝
    for emo, delta in bias.items():
        result[emo] = result.get(emo, 0) + delta

    # 情绪 → 4维反馈
    feedback = emotion_to_persona_feedback(result)
    if "mood" in feedback:
        state.mood = max(0.0, min(100.0, state.mood + feedback["mood"]))
    if "energy" in feedback:
        state.energy = max(0.0, min(100.0, state.energy + feedback["energy"]))
    if "social_need" in feedback:
        state.social_need = max(0.0, min(100.0, state.social_need + feedback["social_need"]))

    logger.debug(
        f"[PersonaBridge] process_interaction scope={scope_id} "
        f"state=(E={state.energy:.0f} M={state.mood:.0f} "
        f"S={state.social_need:.0f} SA={state.satiety:.0f}) "
        f"bias={bias} feedback={feedback}"
    )
    return result
