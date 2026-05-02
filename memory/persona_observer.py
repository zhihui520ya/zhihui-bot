"""
⚠️ 此文件已废弃 — 行为观察功能已合并到 user_profile.py。

曾经的功能（observe_and_update）：通过 LLM 分析用户沟通风格/情绪模式，
写入 profile_json["_observations"]。

当前状态：
- _observations 字段仍被 reply_engine.py 和 admin_panel.py 读取展示
- 但写入逻辑（observe_and_update）从未被接入主流程
- 如需重新启用，在 reply_engine.py 的 send_reply 末尾加上：
    await observe_and_update(user_id, user_msg, reply, llm_for_emotion)
"""
