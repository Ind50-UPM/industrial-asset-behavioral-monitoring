"""Internationalization helpers for the Model_A command-line interface."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable


def setup_i18n(lang: str = "en") -> Callable[[str], str]:
    """Return a translation function for the requested interface language.

    The project stores human-maintained translations in ``locales/*/LC_MESSAGES``
    as ``.po`` files. This helper reads those catalogs directly so the CLI can be
    translated even when ``.mo`` files have not been compiled yet.

    Args:
        lang: ISO language code requested by the user.

    Returns:
        A callable compatible with ``gettext`` usage that translates a message
        identifier into the configured language. English falls back to the
        original message identifiers.
    """
    if lang == "en":
        return lambda message: message

    locale_dir = Path(__file__).resolve().parents[1] / "locales" / lang / "LC_MESSAGES"
    catalog_path = locale_dir / "messages.po"
    catalog = _load_po_catalog(catalog_path)
    return lambda message: catalog.get(message, message)


def _load_po_catalog(path: Path) -> dict[str, str]:
    """Parse a minimal ``.po`` catalog into an in-memory translation mapping.

    The parser intentionally supports the subset of the GNU gettext format used
    by this repository: singular ``msgid`` and ``msgstr`` entries with optional
    multiline string fragments.

    Args:
        path: Path to the ``.po`` file to parse.

    Returns:
        A dictionary keyed by source ``msgid`` values.
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
    """Decode a quoted gettext string fragment into plain Python text.

    Args:
        value: One quoted fragment from a gettext catalog.

    Returns:
        The decoded Unicode string represented by the fragment.
    """
    return ast.literal_eval(value)
