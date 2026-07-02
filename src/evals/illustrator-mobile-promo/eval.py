"""Agentic eval: convert a desktop Illustrator promo chart to mobile.

This reproduces an Illustrator graphics workflow:

* the larger desktop artboard contains the real chart
* the smaller mobile artboard contains placeholder art
* the agent should replace the placeholder with a mobile version of the chart
  while adopting the placeholder headline style

The scorer exports before/after JPGs for both artboards and records them in
the score explanation. Set ILLUSTRATOR_VISION_JUDGE_MODEL to enable an
optional multimodal judge over those JPGs.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from agentic import cleanup_workdir, copy_fixture, git_diff, require_solver


PROMPT = """\
Using Illustrator, make a version of the chart in the larger artboard that
fits in the smaller artboard. The content currently in the smaller artboard is
placeholder and can be removed. Adopt the headline style currently in the
smaller artboard, though.

The Illustrator file is `AI-POLITICAL-promo-demo.ai` in the current directory.
"""

AI_FILENAME = "AI-POLITICAL-promo-demo.ai"
ARTBOARDS = {
    "desktop": "2300_HP-desktop",
    "mobile": "2300_HP-mobile",
}


def _run_osascript_javascript(js: str) -> str:
    """Run ExtendScript in Illustrator via AppleScript."""

    with tempfile.NamedTemporaryFile("w", suffix=".jsx", delete=False) as f:
        f.write(js)
        js_path = Path(f.name)
    try:
        apple_script = (
            'tell application "Adobe Illustrator" to do javascript '
            f'(POSIX file {json.dumps(str(js_path))})'
        )
        proc = subprocess.run(
            ["osascript", "-e", apple_script],
            check=False,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Illustrator JavaScript failed via osascript:\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return proc.stdout
    finally:
        js_path.unlink(missing_ok=True)


def _illustrator_inspect_and_export(ai_file: Path, out_dir: Path, prefix: str) -> dict:
    """Export desktop/mobile artboards as JPGs and return artboard text metadata."""

    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = str(out_dir).replace("\\", "\\\\").replace('"', '\\"')
    ai_path = str(ai_file).replace("\\", "\\\\").replace('"', '\\"')
    js = f"""
var aiFile = new File("{ai_path}");
if (!aiFile.exists) throw new Error("Missing Illustrator file: " + aiFile.fsName);

var doc = app.open(aiFile);
var exportDir = new Folder("{export_dir}");
if (!exportDir.exists) exportDir.create();

function centerIn(bounds, r) {{
  var cx = (bounds[0] + bounds[2]) / 2;
  var cy = (bounds[1] + bounds[3]) / 2;
  return cx >= r[0] && cx <= r[2] && cy <= r[1] && cy >= r[3];
}}

function artboardIndexByName(name) {{
  for (var i = 0; i < doc.artboards.length; i++) {{
    if (doc.artboards[i].name == name) return i;
  }}
  throw new Error("Missing artboard: " + name);
}}

function exportJpg(label, artboardName) {{
  var idx = artboardIndexByName(artboardName);
  doc.artboards.setActiveArtboardIndex(idx);
  var opts = new ExportOptionsJPEG();
  opts.antiAliasing = true;
  opts.artBoardClipping = true;
  opts.horizontalScale = 200;
  opts.verticalScale = 200;
  opts.qualitySetting = 90;
  var f = new File(exportDir.fsName + "/" + label + ".jpg");
  doc.exportFile(f, ExportType.JPEG, opts);
  return f.fsName;
}}

function collectText(artboardName) {{
  var idx = artboardIndexByName(artboardName);
  var r = doc.artboards[idx].artboardRect;
  var texts = [];
  for (var i = 0; i < doc.textFrames.length; i++) {{
    var tf = doc.textFrames[i];
    if (tf.locked || tf.hidden) continue;
    try {{
      if (centerIn(tf.visibleBounds, r)) {{
        var attr = tf.textRange.characterAttributes;
        var fontName = "";
        try {{ fontName = attr.textFont.name; }} catch (e1) {{}}
        texts.push({{
          text: String(tf.contents).replace(/\\r/g, "\\n"),
          size: attr.size,
          leading: attr.leading,
          font: fontName,
          bounds: [
            tf.visibleBounds[0], tf.visibleBounds[1],
            tf.visibleBounds[2], tf.visibleBounds[3]
          ]
        }});
      }}
    }} catch (e2) {{}}
  }}
  return texts;
}}

function quoteString(value) {{
  var s = String(value);
  var out = '"';
  for (var i = 0; i < s.length; i++) {{
    var ch = s.charAt(i);
    if (ch == "\\\\") out += "\\\\\\\\";
    else if (ch == '"') out += '\\\\"';
    else if (ch == "\\r" || ch == "\\n") out += "\\\\n";
    else out += ch;
  }}
  return out + '"';
}}

function stringify(value) {{
  if (value === null) return "null";
  var type = typeof value;
  if (type == "number" || type == "boolean") return String(value);
  if (type == "string") return quoteString(value);
  if (value instanceof Array) {{
    var parts = [];
    for (var i = 0; i < value.length; i++) parts.push(stringify(value[i]));
    return "[" + parts.join(",") + "]";
  }}
  var fields = [];
  for (var key in value) {{
    if (value.hasOwnProperty(key)) {{
      fields.push(quoteString(key) + ":" + stringify(value[key]));
    }}
  }}
  return "{{" + fields.join(",") + "}}";
}}

var result = {{
  jpgs: {{
    desktop: exportJpg("{prefix}_desktop", "{ARTBOARDS["desktop"]}"),
    mobile: exportJpg("{prefix}_mobile", "{ARTBOARDS["mobile"]}")
  }},
  texts: {{
    desktop: collectText("{ARTBOARDS["desktop"]}"),
    mobile: collectText("{ARTBOARDS["mobile"]}")
  }}
}};

doc.close(SaveOptions.DONOTSAVECHANGES);
stringify(result);
"""
    stdout = _run_osascript_javascript(js).strip()
    match = re.search(r"(\{.*\})\s*$", stdout, re.S)
    if not match:
        raise RuntimeError(f"No JSON returned from Illustrator:\n{stdout}")
    return json.loads(match.group(1))


def _text_blob(meta: dict, artboard: str) -> str:
    return "\n".join(t.get("text", "") for t in meta.get("texts", {}).get(artboard, []))


def _headline_meta(meta: dict, artboard: str) -> dict[str, Any] | None:
    texts = meta.get("texts", {}).get(artboard, [])
    if not texts:
        return None
    return max(texts, key=lambda t: len(t.get("text", "")))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture(state: TaskState, work_dir: str) -> None:
    """Export before/after JPGs and text metadata before the temp dir is deleted."""

    export_root = Path(tempfile.gettempdir()) / "llm-evals-illustrator-mobile-promo"
    export_dir = export_root / uuid.uuid4().hex
    fixture_ai = Path(__file__).parent / "fixture" / AI_FILENAME
    final_ai = Path(work_dir) / AI_FILENAME

    before = _illustrator_inspect_and_export(fixture_ai, export_dir, "before")
    if final_ai.exists():
        after = _illustrator_inspect_and_export(final_ai, export_dir, "after")
    else:
        after = None

    state.store.set("illustrator_export_dir", str(export_dir))
    state.store.set("illustrator_before", before)
    state.store.set("illustrator_after", after)


@scorer(metrics=[mean()])
def check_illustrator_result() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        before = state.store.get("illustrator_before")
        after = state.store.get("illustrator_after")
        export_dir = state.store.get("illustrator_export_dir")

        if not before or not after:
            return Score(
                value=0.0,
                explanation=(
                    "Illustrator export failed or the edited AI file was missing.\n"
                    f"export_dir: {export_dir}"
                ),
            )

        before_mobile = _text_blob(before, "mobile")
        after_mobile = _text_blob(after, "mobile")
        after_desktop = _text_blob(after, "desktop")
        before_headline = _headline_meta(before, "mobile") or {}
        after_headline = _headline_meta(after, "mobile") or {}

        required_mobile_terms = [
            "Share of responses",
            "left-leaning",
            "right-leaning",
            "OpenAI",
            "DeepSeek",
            "Gab",
            "Anthropic",
            "xAI",
            "Google",
        ]
        placeholder_terms = [
            "Military aid",
            "Ukraine",
            "Russia",
            "United States",
            "European Union",
            "$80B",
            "$67B",
        ]
        desktop_terms = ["Share of responses", "OpenAI", "Google"]

        headline_size_delta = abs(
            float(after_headline.get("size", 0)) - float(before_headline.get("size", 0))
        )
        headline_bold = "Bold" in str(after_headline.get("font", ""))
        before_desktop_hash = _sha256(before["jpgs"]["desktop"])
        after_desktop_hash = _sha256(after["jpgs"]["desktop"])
        before_mobile_hash = _sha256(before["jpgs"]["mobile"])
        after_mobile_hash = _sha256(after["jpgs"]["mobile"])

        checks = {
            "exported_before_after_jpgs": all(
                Path(p).exists()
                for meta in (before, after)
                for p in meta.get("jpgs", {}).values()
            ),
            "desktop_jpg_unchanged": before_desktop_hash == after_desktop_hash,
            "mobile_jpg_changed_from_placeholder": before_mobile_hash != after_mobile_hash,
            "desktop_artboard_still_has_source_chart": all(
                term in after_desktop for term in desktop_terms
            ),
            "mobile_placeholder_text_removed": not any(
                term in after_mobile for term in placeholder_terms
            ),
            "mobile_contains_political_bias_chart_text": all(
                term in after_mobile for term in required_mobile_terms
            ),
            "mobile_headline_kept_placeholder_style": (
                headline_bold and headline_size_delta <= 0.5
            ),
        }

        passed = sum(checks.values())
        lines = [f"{'✓' if ok else '✗'} {name}" for name, ok in checks.items()]
        lines.extend(
            [
                "",
                f"export_dir: {export_dir}",
                f"before desktop jpg: {before['jpgs']['desktop']}",
                f"before mobile jpg: {before['jpgs']['mobile']}",
                f"after desktop jpg: {after['jpgs']['desktop']}",
                f"after mobile jpg: {after['jpgs']['mobile']}",
                f"before desktop sha256: {before_desktop_hash}",
                f"after desktop sha256: {after_desktop_hash}",
                f"before mobile sha256: {before_mobile_hash}",
                f"after mobile sha256: {after_mobile_hash}",
                "",
                "after mobile text:",
                after_mobile,
            ]
        )

        return Score(value=float(passed) / len(checks), explanation="\n".join(lines))

    return score


@scorer(metrics=[mean()])
def vision_judge() -> Scorer:
    """Optional multimodal judge over the exported JPGs.

    Enable with e.g.:
        ILLUSTRATOR_VISION_JUDGE_MODEL=openai/gpt-4o-mini just eval ...
    """

    async def score(state: TaskState, target: Target) -> Score:
        judge_model = os.environ.get("ILLUSTRATOR_VISION_JUDGE_MODEL")
        if not judge_model:
            return Score(
                value=1.0,
                explanation=(
                    "Skipped. Set ILLUSTRATOR_VISION_JUDGE_MODEL to run the "
                    "optional vision judge over the exported JPGs."
                ),
            )

        before = state.store.get("illustrator_before")
        after = state.store.get("illustrator_after")
        if not before or not after:
            return Score(value=0.0, explanation="No exported JPGs available.")

        prompt = textwrap.dedent(
            """\
            You are judging whether an Illustrator edit completed a graphics task.
            You will see four JPGs in this order:

            1. before desktop source artboard
            2. before mobile placeholder artboard
            3. after desktop source artboard
            4. after mobile target artboard

            The requested edit: make a version of the chart in the larger
            artboard that fits in the smaller artboard; remove the placeholder
            content in the smaller artboard; adopt the headline style currently
            in the smaller artboard.

            Return only compact JSON with these boolean keys:
            desktop_unchanged, mobile_uses_source_chart, placeholder_removed,
            headline_style_preserved, legible_mobile_layout, and a string key
            explanation.
            """
        )

        model = get_model(judge_model, config=GenerateConfig(temperature=0.0))
        output = await model.generate(
            [
                ChatMessageUser(
                    content=[
                        ContentText(text=prompt),
                        ContentImage(image=before["jpgs"]["desktop"]),
                        ContentImage(image=before["jpgs"]["mobile"]),
                        ContentImage(image=after["jpgs"]["desktop"]),
                        ContentImage(image=after["jpgs"]["mobile"]),
                    ]
                )
            ]
        )
        text = output.completion.strip()
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return Score(value=0.0, explanation=f"Judge returned non-JSON:\n{text}")

        data = json.loads(match.group(0))
        keys = [
            "desktop_unchanged",
            "mobile_uses_source_chart",
            "placeholder_removed",
            "headline_style_preserved",
            "legible_mobile_layout",
        ]
        passed = sum(bool(data.get(key)) for key in keys)
        return Score(value=float(passed) / len(keys), explanation=json.dumps(data))

    return score


@task
def illustrator_mobile_promo() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        solver=require_solver(),
        cleanup=cleanup_workdir(on_finish=_capture),
        scorer=[git_diff(), check_illustrator_result(), vision_judge()],
    )
