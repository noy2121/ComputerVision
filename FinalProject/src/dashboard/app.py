import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra
from omegaconf import DictConfig
from dash import Dash, html, dcc, Input, Output

from flask import send_from_directory, abort 

from src.dashboard.tabs.inference import build_inference_tab, register_inference_callbacks
from src.dashboard.tabs.performance import build_performance_tab, register_performance_callbacks
from src.dashboard.tabs.dataset import build_dataset_tab, register_dataset_callbacks

from src.dashboard.tabs.evaluation_plots import (
    build_evaluation_plots_tab,
    register_evaluation_plots_callbacks,
)

from src.pipeline import Pipeline


def create_app(cfg: DictConfig):
    app = Dash(__name__, suppress_callback_exceptions=True)

    pipeline = Pipeline(cfg)
    pipeline.setup()

    plots_dir = (Path(cfg.paths.results) / "classifier_eval_plots").resolve()
    app.server.config["EVAL_PLOTS_DIR"] = str(plots_dir)

    @app.server.route("/eval-plots/<path:filename>")
    def _serve_eval_plots(filename: str):
        base = Path(app.server.config["EVAL_PLOTS_DIR"]).resolve()
        file_path = (base / filename).resolve()

        if base not in file_path.parents and file_path != base:
            abort(404)

        if not file_path.exists():
            abort(404)

        return send_from_directory(base, filename)

    app.layout = html.Div([
        html.H1("Crop Freshness Detection System", style={'textAlign': 'center', 'padding': '20px'}),

        dcc.Tabs(id='main-tabs', value='inference', children=[
            dcc.Tab(label='Live Inference', value='inference'),
            dcc.Tab(label='Model Performance', value='performance'),
            dcc.Tab(label='Evaluation Plots', value='eval_plots'),
            dcc.Tab(label='Dataset Overview', value='dataset'),
        ]),

        html.Div(id='tab-content', style={'padding': '20px'})
    ])

    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_tab(tab):
        if tab == 'inference':
            return build_inference_tab()
        elif tab == 'performance':
            return build_performance_tab()
        elif tab == 'eval_plots':
            return build_evaluation_plots_tab(cfg)
        elif tab == 'dataset':
            return build_dataset_tab(cfg)

    register_inference_callbacks(app, pipeline)
    register_performance_callbacks(app, pipeline)
    register_dataset_callbacks(app)

    register_evaluation_plots_callbacks(app, cfg)

    return app


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    app = create_app(cfg)

    app.run(
        host=cfg.dashboard.host,
        port=cfg.dashboard.port,
        debug=cfg.dashboard.debug
    )


if __name__ == "__main__":
    main()
