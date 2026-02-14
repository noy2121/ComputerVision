from pathlib import Path

from dash import html, dcc
import plotly.graph_objects as go

from src.data.analyze_data import analyze_dataset, create_visualization, create_combined_visualization


def build_dataset_tab(cfg):
    root_dir = Path(cfg.data.root_dir)

    all_stats = {}
    for dataset in cfg.data.datasets:
        if dataset.enabled:
            dataset_path = root_dir / dataset.name
            stats = analyze_dataset(dataset_path)
            all_stats[dataset.name] = stats

    total_images = sum(sum(s['count'] for s in stats.values()) for stats in all_stats.values())
    total_classes = sum(len(stats) for stats in all_stats.values())

    children = [
        html.H3("Dataset Overview"),

        html.Div([
            html.Div([
                html.H4(f"{total_images:,}", style={'fontSize': '36px', 'margin': '5px'}),
                html.P("Total Images")
            ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),

            html.Div([
                html.H4(f"{total_classes}", style={'fontSize': '36px', 'margin': '5px'}),
                html.P("Total Classes")
            ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),

            html.Div([
                html.H4(f"{len(all_stats)}", style={'fontSize': '36px', 'margin': '5px'}),
                html.P("Datasets")
            ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),

            html.Div([
                html.H4(f"{cfg.data.train_split:.0%} / {cfg.data.val_split:.0%} / {cfg.data.test_split:.0%}",
                         style={'fontSize': '24px', 'margin': '5px'}),
                html.P("Train / Val / Test")
            ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px',
                  'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
    ]

    for dataset_name, stats in all_stats.items():
        fig = create_visualization(dataset_name, stats)
        children.append(dcc.Graph(figure=fig))

    if len(all_stats) > 1:
        combined_fig = create_combined_visualization(all_stats)
        children.append(dcc.Graph(figure=combined_fig))

    header = html.Tr([html.Th('Class'), html.Th('Count'), html.Th('Dataset')])
    rows = []
    for dataset_name, stats in all_stats.items():
        for class_name, data in sorted(stats.items()):
            color = '#d4edda' if 'Healthy' in class_name else '#f8d7da'
            rows.append(html.Tr([
                html.Td(class_name),
                html.Td(f"{data['count']:,}"),
                html.Td(dataset_name),
            ], style={'backgroundColor': color}))

    children.append(html.H4("Class Details", style={'marginTop': '20px'}))
    children.append(html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={'width': '100%', 'borderCollapse': 'collapse',
               'fontSize': '13px', 'textAlign': 'center'}
    ))

    return html.Div(children)


def register_dataset_callbacks(app):
    pass
