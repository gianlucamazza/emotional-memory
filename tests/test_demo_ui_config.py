"""Static checks for the Gradio 6 demo wiring."""

from __future__ import annotations

import ast
from pathlib import Path


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def _load_demo_tree() -> ast.Module:
    app_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    return ast.parse(app_path.read_text(encoding="utf-8"))


def test_blocks_constructor_does_not_receive_theme() -> None:
    tree = _load_demo_tree()
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


def test_launch_receives_theme_explicitly() -> None:
    tree = _load_demo_tree()
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]

    assert launch_calls
    launch_keywords = _keyword_map(launch_calls[-1])

    assert isinstance(launch_keywords.get("theme"), ast.Name)
    assert launch_keywords["theme"].id == "_DEMO_THEME"


def test_chatbot_explicitly_disables_tags() -> None:
    tree = _load_demo_tree()
    chatbot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Chatbot"
    ]

    assert chatbot_calls
    chatbot_keywords = _keyword_map(chatbot_calls[0])

    assert "type" not in chatbot_keywords
    assert isinstance(chatbot_keywords.get("allow_tags"), ast.Constant)
    assert chatbot_keywords["allow_tags"].value is False
