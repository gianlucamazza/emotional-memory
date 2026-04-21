"""Static checks for the Gradio demo wiring."""

from __future__ import annotations

import ast
from pathlib import Path


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def test_blocks_theme_is_not_passed_in_constructor() -> None:
    app_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    blocks_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Blocks"
    ]

    assert blocks_calls
    for call in blocks_calls:
        assert "theme" not in _keyword_map(call)


def test_launch_receives_theme_and_chatbot_disables_tags() -> None:
    app_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]
    chatbot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Chatbot"
    ]

    assert launch_calls
    assert chatbot_calls

    launch_keywords = _keyword_map(launch_calls[-1])
    chatbot_keywords = _keyword_map(chatbot_calls[0])

    assert isinstance(launch_keywords.get("theme"), ast.Name)
    assert launch_keywords["theme"].id == "_DEMO_THEME"
    assert isinstance(chatbot_keywords.get("allow_tags"), ast.Constant)
    assert chatbot_keywords["allow_tags"].value is False
