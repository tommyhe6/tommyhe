import json
from pathlib import Path


directory = Path(__file__).resolve().parent
data = {
    "demo": json.loads((directory / "asset" / "demo.json").read_text(encoding="utf-8")),
    "trajectories": json.loads((directory / "asset" / "trajectories.json").read_text(encoding="utf-8")),
}
(directory / "data.js").write_text(
    "const contextAnswerData = "
    + json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    + ";\n",
    encoding="utf-8",
)
