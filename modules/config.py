"""Small configuration helpers that fail gracefully when no secrets file exists."""

from __future__ import annotations

from typing import Any

import streamlit as st


def get_secret(name: str, default: Any = None) -> Any:
    """Return a Streamlit secret without crashing when secrets are not configured."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def configured_secret_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return the subset of secret names that currently have non-empty values."""
    return [name for name in names if get_secret(name)]


def missing_secret_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return secret names that are absent or empty."""
    return [name for name in names if not get_secret(name)]
