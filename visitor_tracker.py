from __future__ import annotations

import hashlib
import threading

import requests
import streamlit as st


def _tracking_config() -> dict[str, str] | None:
    """Read the public app's tracking settings without breaking the app."""
    try:
        section = st.secrets["visitor_tracking"]
        names = ("supabase_url", "supabase_publishable_key", "visitor_salt")
        values = {name: str(section[name]).strip() for name in names}
    except Exception:
        return None

    if any(not value for value in values.values()):
        return None
    return values


def _anonymous_visitor_id(visitor_salt: str) -> str:
    """Hash connection data so raw IP and browser information are never saved."""
    ip_address = str(st.context.ip_address or "unknown-ip")
    user_agent = str(st.context.headers.get("User-Agent", "unknown-agent"))
    source = f"{visitor_salt}|{ip_address}|{user_agent}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _send_visit(supabase_url: str, publishable_key: str, visitor_id: str) -> None:
    try:
        response = requests.post(
            f"{supabase_url.rstrip('/')}/rest/v1/rpc/record_daily_visit",
            headers={
                "apikey": publishable_key,
                "Content-Type": "application/json",
            },
            json={"p_visitor_id": visitor_id},
            timeout=(2, 4),
        )
        response.raise_for_status()
    except requests.RequestException:
        # 방문 통계 문제로 공개 페이지가 멈추지 않도록 조용히 건너뜁니다.
        return


def record_visit_once() -> None:
    """Record one visit per browser session without delaying page rendering."""
    config = _tracking_config()
    if config is None or st.session_state.get("visitor_tracking_started"):
        return

    visitor_id = _anonymous_visitor_id(config["visitor_salt"])
    st.session_state["visitor_tracking_started"] = True
    threading.Thread(
        target=_send_visit,
        args=(
            config["supabase_url"],
            config["supabase_publishable_key"],
            visitor_id,
        ),
        daemon=True,
    ).start()
