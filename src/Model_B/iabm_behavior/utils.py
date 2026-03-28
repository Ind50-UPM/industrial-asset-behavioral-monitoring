"""Internationalization helpers for the Model_B command-line interface."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable


def setup_i18n(lang: str = "en") -> Callable[[str], str]:
    """Return a translation function for the requested interface language.

    Args:
        lang: ISO language code requested by the caller.

    Returns:
        A callable that translates user-facing CLI strings into the requested
        language, falling back to the original message when no translation
        exists.
    """
    if lang == "en":
        return lambda message: message

    catalog_path = (
        Path(__file__).resolve().parents[1] / "locales" / lang / "LC_MESSAGES" / "messages.po"
    )
    catalog = _load_po_catalog(catalog_path)
    return lambda message: catalog.get(message, message)


def _load_po_catalog(path: Path) -> dict[str, str]:
    """Parse a minimal gettext catalog into a dictionary.

    Args:
        path: Path to the ``.po`` file to parse.

    Returns:
        A dictionary mapping ``msgid`` values to translated strings.
    """
    if not path.exists():
        return {}

    catalog: dict[str, str] = {}
    current_field: str | None = None
    current_msgid: list[str] = []
    current_msgstr: list[str] = []

    def flush_entry() -> None:
        if not current_msgid:
            return
        msgid = "".join(current_msgid)
        msgstr = "".join(current_msgstr)
        if msgid:
            catalog[msgid] = msgstr or msgid

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("msgid "):
            flush_entry()
            current_msgid = [_decode_po_string(line[6:])]
            current_msgstr = []
            current_field = "msgid"
            continue
        if line.startswith("msgstr "):
            current_msgstr = [_decode_po_string(line[7:])]
            current_field = "msgstr"
            continue
        if line.startswith('"'):
            if current_field == "msgid":
                current_msgid.append(_decode_po_string(line))
            elif current_field == "msgstr":
                current_msgstr.append(_decode_po_string(line))

    flush_entry()
    return catalog


def _decode_po_string(value: str) -> str:
    """Decode a quoted gettext fragment into plain Python text.

    Args:
        value: One quoted fragment from a gettext catalog.

    Returns:
        The decoded Unicode string represented by the fragment.
    """
    return ast.literal_eval(value)
