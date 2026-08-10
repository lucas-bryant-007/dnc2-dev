import ast
import json
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json"}


def test_no_live_debugger_calls():
    offenders = []
    for path in Path(".").rglob("*.py"):
        if any(part.startswith(".") or part == "repro_exports" for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "breakpoint":
                    offenders.append(str(path))
    assert offenders == []


def test_current_text_tree_has_no_conversation_artifact_terms():
    forbidden = (
        "trans" + "cript",
        "trans" + "cription",
        "trans" + "cribed",
        "chat" + "gpt",
        "clau" + "de",
        "cur" + "sor",
    )
    offenders = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".pytest_cache", ".ruff_cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(term in text for term in forbidden):
            offenders.append(str(path))
    assert offenders == []


def test_saved_metric_schema_uses_correct_labels():
    for path in Path("metrics").glob("ro3*.json"):
        text = path.read_text(encoding="utf-8")
        assert '"obs_se"' not in text
        assert '"cap_se"' not in text
        assert '"goal coverage"' not in text
    for path in Path("metrics").glob("hyperrect_bounds*.json"):
        text = path.read_text(encoding="utf-8")
        assert '"observed_side"' not in text
        assert '"sqrtB_predicted"' not in text


def test_all_saved_metric_json_is_valid():
    for path in Path("metrics").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
