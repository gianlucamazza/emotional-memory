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


def test_launch_uses_explicit_launch_kwargs_helper() -> None:
    tree = _load_demo_tree()
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]

    assert launch_calls
    launch_call = launch_calls[-1]

    assert not launch_call.args
    assert len(launch_call.keywords) == 1
    keyword = launch_call.keywords[0]
    assert keyword.arg is None
    assert isinstance(keyword.value, ast.Call)
    assert isinstance(keyword.value.func, ast.Name)
    assert keyword.value.func.id == "_launch_kwargs"


def test_launch_receives_explicit_ssr_mode() -> None:
    tree = _load_demo_tree()
    launch_helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_kwargs"
    ]

    assert launch_helpers
    helper = launch_helpers[0]
    returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]

    assert returns
    assert isinstance(returns[0].value, ast.Dict)

    key_values = {
        key.value: value
        for key, value in zip(returns[0].value.keys, returns[0].value.values, strict=False)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert "ssr_mode" in key_values
    assert isinstance(key_values["ssr_mode"], ast.Call)
    assert isinstance(key_values["ssr_mode"].func, ast.Name)
    assert key_values["ssr_mode"].func.id == "_should_enable_ssr"


def test_should_enable_ssr_reads_only_demo_env_flag() -> None:
    tree = _load_demo_tree()
    ssr_helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_should_enable_ssr"
    ]

    assert ssr_helpers
    helper = ssr_helpers[0]
    string_constants = {
        node.value for node in ast.walk(helper) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "EMOTIONAL_MEMORY_DEMO_SSR" in string_constants
    assert "GRADIO_SSR_MODE" not in string_constants


def test_event_loop_cleanup_patch_runs_before_gradio_import() -> None:
    tree = _load_demo_tree()
    body = tree.body

    patch_call_index = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_patch_event_loop_cleanup"
    )
    gradio_import_index = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Import)
        and any(alias.name == "gradio" for alias in node.names)
    )

    assert patch_call_index < gradio_import_index


def test_event_loop_cleanup_patch_is_narrow_and_idempotent() -> None:
    tree = _load_demo_tree()
    helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_event_loop_cleanup"
    ]

    assert helpers
    helper = helpers[0]
    source = ast.unparse(helper)

    assert "BaseEventLoop.__del__" in source
    assert "__emotional_memory_patched__" in source
    assert "ValueError" in source
    assert "Invalid file descriptor" in source


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
