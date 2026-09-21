"""Tests for the optional ``window_limit`` on ``load_messages``.

The hot restore path bounds the SQL SELECT to a row window so a long
session history is not pulled back in full on every session restore,
while the default (``window_limit=0``) keeps the exact historical
behaviour: all rows are scanned and turns are trimmed in Python.
"""

from agent.memory import clear_conversation_store_cache, get_conversation_store


def _make_store(tmp_path):
    clear_conversation_store_cache()
    return get_conversation_store(str(tmp_path / "memory" / "long-term"))


def _seed_turns(store, session_id, n_turns):
    """Append n user/assistant turn pairs, incl. a tool_use + tool_result."""
    messages = []
    for i in range(n_turns):
        messages.append({"role": "user", "content": [{"type": "text", "text": f"user {i}"}]})
        if i % 3 == 0:
            messages.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "f", "input": {}}],
            })
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}],
            })
        messages.append({"role": "assistant", "content": f"reply {i}"})


def test_default_no_window_matches_historical_call(tmp_path):
    store = _make_store(tmp_path)
    _seed_turns(store, "s1", 40)
    store.append_messages("s1", [
        {"role": "user", "content": [{"type": "text", "text": "final"}]},
        {"role": "assistant", "content": "done"},
    ], channel_type="web")

    explicit = store.load_messages("s1", max_turns=5)
    implicit = store.load_messages("s1", max_turns=5, window_limit=0)
    assert explicit == implicit


def test_window_gives_same_result_with_enough_headroom(tmp_path):
    store = _make_store(tmp_path)
    _seed_turns(store, "s1", 40)
    store.append_messages("s1", [
        {"role": "user", "content": [{"type": "text", "text": "final"}]},
        {"role": "assistant", "content": "done"},
    ], channel_type="web")

    full = store.load_messages("s1", max_turns=5)
    windowed = store.load_messages("s1", max_turns=5, window_limit=30)
    assert windowed == full


def test_window_caps_rows_returned(tmp_path):
    store = _make_store(tmp_path)
    _seed_turns(store, "s1", 40)
    store.append_messages("s1", [
        {"role": "user", "content": [{"type": "text", "text": "final"}]},
        {"role": "assistant", "content": "done"},
    ], channel_type="web")

    # A tiny window forces the SQL LIMIT to bind and can cut mid-turn, so the
    # result is intentionally *not* a full conversation — it must, however,
    # only ever contain rows from the recent tail.
    small = store.load_messages("s1", max_turns=5, window_limit=3)
    assert len(small) <= 3

    # The last visible user message of the full window is still the newest one.
    full_last_user = next(m for m in reversed(store.load_messages("s1", max_turns=5))
                          if m["role"] == "user")
    small_last_user = next(m for m in reversed(small) if m["role"] == "user")
    assert small_last_user == full_last_user