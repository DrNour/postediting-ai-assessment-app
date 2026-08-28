"""Shared teacher-access gate for Streamlit pages."""

from __future__ import annotations

import hmac

import streamlit as st

from modules.config import get_secret


_SESSION_KEY = "teacher_logged_in"


def require_teacher_access(page_key: str) -> bool:
    """Render a shared sidebar login and return whether teacher access is granted."""
    configured_password = get_secret("TEACHER_PASSWORD")

    if not configured_password:
        st.error(
            "TEACHER_PASSWORD is not configured. Copy "
            "`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
            "and add a strong password."
        )
        return False

    if st.session_state.get(_SESSION_KEY, False):
        st.sidebar.success("Teacher access granted")
        if st.sidebar.button("Log out", key=f"teacher_logout_{page_key}"):
            st.session_state[_SESSION_KEY] = False
            st.rerun()
        return True

    st.sidebar.subheader("Teacher access")
    entered_password = st.sidebar.text_input(
        "Teacher password",
        type="password",
        key=f"teacher_password_{page_key}",
    )

    if st.sidebar.button("Unlock teacher tools", key=f"teacher_login_{page_key}"):
        if hmac.compare_digest(str(entered_password), str(configured_password)):
            st.session_state[_SESSION_KEY] = True
            st.rerun()
        else:
            st.sidebar.error("Incorrect password.")

    st.warning("This page is restricted to the instructor.")
    return False
