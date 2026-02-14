import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra
from omegaconf import DictConfig
from dash import Dash, html, dcc, Input, Output

from src.dashboard.tabs.inference import build_inference_tab, register_inference_callbacks
from src.dashboard.tabs.performance import build_performance_tab, register_performance_callbacks
from src.dashboard.tabs.dataset import build_dataset_tab, register_dataset_callbacks
from src.pipeline import Pipeline


def create_app(cfg: DictConfig):
    app = Dash(__name__, suppress_callback_exceptions=True)

    pipeline = Pipeline(cfg)
    pipeline.setup()

    app.layout = html.Div([
        html.H1("Crop Freshness Detection System", style={'textAlign': 'center', 'padding': '20px'}),

        dcc.Tabs(id='main-tabs', value='inference', children=[
            dcc.Tab(label='Live Inference', value='inference'),
            dcc.Tab(label='Model Performance', value='performance'),
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
        elif tab == 'dataset':
            return build_dataset_tab(cfg)

    register_inference_callbacks(app, pipeline)
    register_performance_callbacks(app, pipeline)
    register_dataset_callbacks(app)

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
