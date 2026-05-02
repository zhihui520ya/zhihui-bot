"""
FastJudge 轻量级预判层

在调用主模型之前，用廉价模型快速判断：
  1. 是否需要回复（score 0-10）
  2. 是否强制回复（@bot/私聊提问）
  3. 是否强制跳过（广告/复读）
  4. 话题关键词 + 语气建议 + 一句话理由

预判异常时返回 None，调用方自行兜底（回退到原概率逻辑）。

模型：deepseek-v4-flash（廉价），与主模型 deepseek-reasoner 分离。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("qq-bot")

# 预判模型名称
JUDGE_MODEL = "deepseek-v4-flash"


@dataclass
class FastJudgeDecision:
    """快速预判结果"""
    score: int = 5               # 0-10，≥7 强烈建议回复，≤3 建议忽略
    force_reply: bool = False    # 强制回复（@bot、私聊提问等）
    force_skip: bool = False     # 强制跳过（广告刷屏、复读等）
    topic: str = ""              # 1-3 词话题标签
    tone_hint: str = "normal"    # 语气建议：normal / happy / comfort / tease / angry / short
    reasoning: str = ""          # 一句话理由（给主模型看，15 字内最佳）


# 简易指令（单条 HumanMessage，不用 SystemMessage 避免额外开销）
_JUDGE_INSTRUCTION = (
    "分析以下消息，判断是否需要回复（输出纯 JSON，不要其他内容）：\n\n"
    "规则：\n"
    "- 用户在提问、@bot、表达强烈情绪 → 必须回复\n"
    "- 私聊消息基本都需要回复（除非广告/骚扰）\n"
    "- 群聊日常闲聊可忽略\n"
    "- 群聊中有人明确与 bot 对话（提问/吐槽/命令）→ 回复\n\n"
    'JSON 格式：\n'
    '{\n'
    '  "score": <0-10 整数, 回复意愿/必要性>,\n'
    '  "force_reply": <true/false, 是否绕过概率判断强制回复>,\n'
    '  "force_skip": <true/false, 是否强制跳过>,\n'
    '  "topic": "<1-3个词概括话题>",\n'
    '  "tone_hint": "<normal|happy|comfort|tease|angry|short>",\n'
    '  "reasoning": "<一句提示，10字内>"\n'
    '}'
)

_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        from llm_factory import get_llm
        _judge_llm = get_llm(JUDGE_MODEL, temperature=0.1, timeout=15, max_retries=1)
    return _judge_llm


async def fast_judge(
    text: str,
    message_type: str,
    emotion_summary: str = "",
) -> Optional[FastJudgeDecision]:
    """快速预判：是否需要回复。

    Args:
        text: 用户消息原文
        message_type: "group" 或 "private"
        emotion_summary: AI 当前情绪摘要（可选），如 "AI当前: 开心(6) 撒娇(3)"

    Returns:
        FastJudgeDecision 或 None（预判失败时）
    """
    if not text or not text.strip():
        return None

    scope_label = "私聊" if message_type == "private" else "群聊"
    emo_line = f"\n{emotion_summary}" if emotion_summary else ""
    prompt = f"{_JUDGE_INSTRUCTION}\n\n消息类型：{scope_label}\n用户消息：{text.strip()[:300]}{emo_line}"

    llm = _get_judge_llm()
    try:
        from langchain_core.messages import HumanMessage
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        content = resp.content.strip()

        # 提取 JSON（兼容 LLM 输出前后的额外文本）
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            logger.debug(f"[FastJudge] 无有效JSON: {content[:80]}")
            return None

        data = json.loads(content[start:end + 1])
        score = max(0, min(10, int(data.get("score", 5))))
        force_reply = bool(data.get("force_reply", False))
        force_skip = bool(data.get("force_skip", False))
        topic = str(data.get("topic", ""))[:30]
        tone_hint = str(data.get("tone_hint", "normal"))[:10]
        reasoning = str(data.get("reasoning", ""))[:50]

        # 私聊消息：只要不是明显可忽略（score >= 3），强制回复
        if message_type == "private" and score >= 3:
            force_reply = True

        logger.debug(
            f"[FastJudge] scope={scope_label} score={score} "
            f"force_reply={force_reply} force_skip={force_skip} "
            f"topic={topic} tone={tone_hint} reason={reasoning}"
        )
        return FastJudgeDecision(
            score=score, force_reply=force_reply, force_skip=force_skip,
            topic=topic, tone_hint=tone_hint, reasoning=reasoning,
        )

    except Exception as e:
        logger.debug(f"[FastJudge] 预判失败（回退到概率逻辑）: {e}")
        return None
