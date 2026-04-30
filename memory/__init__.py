# memory/__init__.py
from .chroma_store import get_vector_store
from .emotion_store import (
    get_session_emotion,
    set_session_emotion,
    clear_session_emotion,
    update_session_emotion,
    get_user_affection,
    update_user_affection,
    get_last_message_time,
    update_last_message_time,
    has_anchor_label,
    get_anchored_labels,
    record_anchor_label,
    is_anchor_in_cooldown,
    record_anchor_resurrect,
)
from .user_profile import (
    get_profile,
    update_profile,
    delete_profile,
    record_message,
    extract_and_update,
)