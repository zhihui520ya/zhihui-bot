# agent.py
import re
import threading
from dotenv import load_dotenv
load_dotenv()  # 确保独立导入时也能拿到 API key
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from llm_factory import get_llm
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.weather import get_weather, get_news
from tools.web import web_search, web_fetch
from tools.image import reverse_image_search
from tools.music import play_music
from tools.mail import send_email, check_emails, read_email
from tools.qzone import (
    qzone_view_feeds, qzone_search_post,
    qzone_like_post, qzone_comment_post,
    qzone_publish_post, qzone_view_visitor, qzone_lookup_member,
    qzone_delete_post,
)
from tools.bilibili import search_bilibili_video, get_bilibili_hot, get_video_info
from tools.meme import send_meme, search_meme


# 全局 LLM 实例（线程安全，可共享）
# 高配模型：deepseek-reasoner（开思考模式），用于主对话回复和工具调用
llm = get_llm(model="deepseek-reasoner", max_tokens=8192,
                 timeout=60, max_retries=2)

# 工具列表（线程安全，可共享）
tools = [
    get_weather, get_news,
    web_search, web_fetch, reverse_image_search, play_music, send_email,
    check_emails, read_email,
    qzone_view_feeds, qzone_search_post,
    qzone_like_post, qzone_comment_post,
    qzone_publish_post, qzone_view_visitor, qzone_lookup_member,
    qzone_delete_post,
    search_bilibili_video, get_bilibili_hot, get_video_info,
    send_meme, search_meme,
]

# ========== 默认 system message（模块加载时读取） ==========
_default_system = None


def _read_personality() -> str:
    """读取 personality.txt，缓存到模块变量"""
    global _default_system
    if _default_system is not None:
        return _default_system
    try:
        with open("personality.txt", "r", encoding="utf-8") as f:
            _default_system = f.read().strip()
    except FileNotFoundError:
        _default_system = (
            "你是一个智能助手，可以调用工具来获取信息。如果需要调用工具，"
            "请先告诉用户你要调用什么工具，然后再执行调用。"
        )
        print("未找到 personality.txt，使用默认提示词。")
    return _default_system


# ========== Prompt 模板（不可变，可共享） ==========
annotation = (
    "\n\n[系统内部标注] 你当前的情绪状态: {emotion_context}\n"
    "你不需要复述这段标注，只需让它影响你的语气和用词。"
)
_prompt_template = ChatPromptTemplate.from_messages([
    ("system", _read_personality() + annotation),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# ========== 会话级 AgentExecutor ==========
# 每个 session_id 独立 executor + 锁，不同会话可并发执行
# 上限 100 个，超限时丢弃最旧的
_session_executors: dict[str, AgentExecutor] = {}
_session_locks: dict[str, threading.Lock] = {}
_session_guard = threading.Lock()
_MAX_SESSIONS = 100


def _get_session_executor(session_id: str) -> tuple[AgentExecutor, threading.Lock]:
    """获取会话专属的 AgentExecutor 和锁（懒创建），超过上限时丢弃最旧会话"""
    if session_id not in _session_executors:
        with _session_guard:
            if session_id not in _session_executors:
                # 超过上限时清空（丢弃最旧的）
                if len(_session_executors) >= _MAX_SESSIONS:
                    _session_executors.clear()
                    _session_locks.clear()
                agent = create_tool_calling_agent(llm, tools, _prompt_template)
                executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iterations=3,
                    return_intermediate_steps=True,
                )
                _session_executors[session_id] = executor
                _session_locks[session_id] = threading.Lock()
    return _session_executors[session_id], _session_locks[session_id]


def _extract_user_id(inputs: dict) -> str:
    """从 inputs 的 emotion_context 中提取当前用户的 QQ 号"""
    emotion_ctx = inputs.get("emotion_context", "")
    m = re.search(r'QQ号是 (\d+)', emotion_ctx)
    return m.group(1) if m else ""


def invoke_agent(inputs: dict, session_id: str = "") -> dict:
    """按 session 独立加锁调用 agent_executor.invoke，不同 session 可并发。"""
    _uid = _extract_user_id(inputs)
    _uid_str = f"user_id='{_uid}'" if _uid else "（使用当前对话用户的QQ号）"

    executor, lock = _get_session_executor(session_id)
    with lock:
        result = executor.invoke(inputs)

    user_input = inputs.get("input", "")
    _called_tools = len(result.get("intermediate_steps", []))

    # 检测用户要求了 QZone 实际写操作（评论/点赞/发说说）
    _write_keywords = ["评论", "点赞", "赞", "发说说", "发空间", "发布", "删掉", "删除", "删说说"]
    _read_keywords = ["说说", "空间", "访客", "空间"]
    _asks_write = any(kw in user_input for kw in _write_keywords)
    # 额外匹配 "发" + 引号内容
    if not _asks_write:
        _asks_write = bool(re.search(r'发[（(（:：]?\s*[""「]', user_input))
    _asks_read = any(kw in user_input for kw in _read_keywords)

    # 要求写入操作但没调任何工具 → 重试
    if _asks_write and _called_tools == 0:
        print(f"[重试] 用户要求写入操作但未调工具: {user_input[:40]}...")
        retry_input = (
            f"{user_input}\n\n"
            f"[系统强制指令] 你必须调用对应的 QZone 工具来实际执行操作：\n"
            f"- 评论: qzone_comment_post(content='你的评论', {_uid_str}, pos=0)\n"
            f"- 点赞: qzone_like_post({_uid_str}, ...)\n"
            f"- 删说说: qzone_delete_post(tid='...', feeds_type=1)\n"
            f"- 发说说: qzone_publish_post(content='内容')\n"
            f"不允许口头答应。不允许只调用 qzone_view_feeds 查看而不执行实际操作。\n"
            f"这是强制要求，请立即调用对应的工具。"
        )
        retry_inputs = dict(inputs)
        retry_inputs["input"] = retry_input
        with lock:
            result = executor.invoke(retry_inputs)

        # 第二次还是没调工具 → 再强制重试一次
        _called_tools2 = len(result.get("intermediate_steps", []))
        if _called_tools2 == 0:
            print(f"[重试] 第二次仍未调工具，强制重试: {user_input[:40]}...")
            retry_input2 = (
                f"{user_input}\n\n"
                f"[最终强制指令] 这是最后一次提醒，你必须调用工具：\n"
                f"qzone_comment_post(content='填写评论内容', {_uid_str}, pos=0)\n"
                f"不要再说话，不要询问，立刻调用。"
            )
            retry_inputs2 = dict(inputs)
            retry_inputs2["input"] = retry_input2
            with lock:
                result = executor.invoke(retry_inputs2)
    elif _asks_read and _called_tools == 0:
        print(f"[重试] 用户要求查看操作但未调工具: {user_input[:40]}...")
        retry_input = (
            f"{user_input}\n\n"
            f"[系统提示] 用户要求的是实际操作，请调用对应的工具查看。"
        )
        retry_inputs = dict(inputs)
        retry_inputs["input"] = retry_input
        with lock:
            result = executor.invoke(retry_inputs)

    # 捕获 Agent 错误，让 LLM 生成拟人化回复
    _output = result.get("output", "")
    if _output and ("Agent stopped" in _output or "max iterations" in _output.lower()):
        print(f"[Agent] 工具执行异常，由 LLM 重新生成回复")
        try:
            _msgs = [{"role": "system", "content": _default_system or ""}]
            _chat_history = inputs.get("chat_history", [])
            if isinstance(_chat_history, list):
                for _msg in _chat_history[-4:]:
                    if hasattr(_msg, "type"):
                        _role = "assistant" if _msg.type == "ai" else "user"
                        _msgs.append({"role": _role, "content": _msg.content})
            _msgs.append({
                "role": "user",
                "content": (
                    f"{user_input}\n\n"
                    f"[系统提示] 刚才搜索时遇到了一点小问题，暂时查不到外部信息。"
                    f"用知慧的风格自然地告诉用户，不要提技术术语。"
                ),
            })
            _resp = llm.invoke(_msgs)
            if _resp and _resp.content:
                result["output"] = _resp.content
        except Exception as _e:
            print(f"[Agent] LLM 重写失败: {_e}")

    return result
