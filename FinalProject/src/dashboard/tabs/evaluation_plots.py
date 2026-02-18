from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from dash import html, dcc, Input, Output


MAIN_PLOTS: List[Tuple[str, str]] = [
    ("Confusion Matrix (Counts)", "confusion_matrix_counts.png"),
    ("Confusion Matrix (Normalized)", "confusion_matrix_normalized.png"),
    ("Per-Class Precision", "per_class_precision.png"),
    ("Per-Class Recall", "per_class_recall.png"),
    ("Per-Class F1", "per_class_f1.png"),
]


def _safe_mtime(p: Path) -> int:
    try:
        return int(p.stat().st_mtime)
    except Exception:
        return 0


def _img_url(rel_path: str, base_dir: Path) -> str:
    file_path = base_dir / rel_path
    v = _safe_mtime(file_path)
    return f"/eval-plots/{rel_path}?v={v}"


def build_evaluation_plots_tab(cfg) -> html.Div:
    plots_dir = Path(cfg.paths.results) / "classifier_eval_plots"
    per_class_dir = plots_dir / "per_class_top_confusions"

    options = []
    if per_class_dir.exists():
        files = sorted(per_class_dir.glob("top_confusions__*.png"))
        for f in files:
            cls_name = f.stem.replace("top_confusions__", "")
            options.append({"label": cls_name, "value": f.name})

    default_value = options[0]["value"] if options else None

    return html.Div(
        [
            html.H3("Evaluation Plots"),
            html.Div(f"Loaded from: {plots_dir}", style={"color": "#555", "marginBottom": "16px"}),

            html.Div(
                [
                    html.Div(
                        [
                            html.H5(title),
                            html.Img(
                                src=_img_url(fname, plots_dir),
                                style={"maxWidth": "100%", "border": "1px solid #eee"},
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginBottom": "24px"},
                    )
                    for title, fname in MAIN_PLOTS
                ],
                style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"},
            ),

            html.Hr(),

            html.H4("Per-Class Top Confusions"),
            dcc.Dropdown(
                id="evalplots-class-dd",
                options=options,
                value=default_value,
                placeholder="Select a class...",
                style={"width": "520px"},
                clearable=False,
            ),
            html.Div(style={"height": "12px"}),

            html.Img(
                id="evalplots-topconf-img",
                src=(_img_url(f"per_class_top_confusions/{default_value}", plots_dir) if default_value else ""),
                style={"maxWidth": "1000px", "width": "100%", "border": "1px solid #eee"},
            ),

            html.Div(id="evalplots-debug", style={"display": "none"}),
        ]
    )


def register_evaluation_plots_callbacks(app, cfg) -> None:
    plots_dir = Path(cfg.paths.results) / "classifier_eval_plots"

    @app.callback(
        Output("evalplots-topconf-img", "src"),
        Input("evalplots-class-dd", "value"),
    )
    def _update_top_confusion_image(filename: str):
        if not filename:
            return ""
        rel = f"per_class_top_confusions/{filename}"
        return _img_url(rel, plots_dir)
