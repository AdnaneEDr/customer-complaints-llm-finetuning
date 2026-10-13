"""Create publication source archive without raw narratives or trained weights."""

import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root.parent / "complaints-source.zip"
excluded = {"data", "artifacts", ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(
            part in excluded or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if path.name.startswith("predictions_") or path.name in {"last_inference.json", "ERROR_EXAMPLES.md"}:
            continue
        archive.write(path, relative.as_posix())
print(destination)
