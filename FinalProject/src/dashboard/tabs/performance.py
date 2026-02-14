import numpy as np
import torch
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def build_performance_tab():
    return html.Div([
        html.H3("Model Performance"),

        html.Button("Run Evaluation on Test Set", id='run-evaluation-btn', n_clicks=0,
                    style={'padding': '10px 20px', 'fontSize': '16px', 'marginBottom': '20px'}),

        dcc.Loading(id='loading-evaluation', children=[
            html.Div(id='evaluation-status', style={'padding': '10px'}),

            html.Div([
                html.Div([
                    html.H4("Confusion Matrix"),
                    dcc.Graph(id='confusion-matrix-chart')
                ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("Per-Class Metrics"),
                    html.Div(id='class-metrics-table')
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'overflowY': 'auto', 'maxHeight': '600px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        ], type='circle')
    ])


def register_performance_callbacks(app, pipeline):

    @app.callback(
        [Output('confusion-matrix-chart', 'figure'),
         Output('class-metrics-table', 'children'),
         Output('evaluation-status', 'children')],
        Input('run-evaluation-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def run_evaluation(n_clicks):
        if n_clicks == 0:
            return go.Figure(), html.Div(), ""

        classifier = pipeline.classifier
        idx_to_class = pipeline.idx_to_class

        if classifier.test_loader is None:
            classifier.setup_data()
            classifier.setup_training()

        all_preds = []
        all_labels = []

        classifier.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(classifier.test_loader, desc='Evaluating'):
                images = images.to(classifier.device)
                outputs = classifier.model(images)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)

        present_classes = sorted(set(all_labels.tolist() + all_preds.tolist()))
        class_names = [idx_to_class.get(i, f"class_{i}") for i in present_classes]

        cm = confusion_matrix(all_labels, all_preds, labels=present_classes)

        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={'size': 9},
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        cm_fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='True',
            xaxis_tickangle=-45,
            height=600,
            margin=dict(l=10, r=10, t=10, b=100)
        )

        report = classification_report(
            all_labels, all_preds,
            labels=present_classes,
            target_names=class_names,
            output_dict=True
        )

        header = html.Tr([
            html.Th('Class'), html.Th('Precision'),
            html.Th('Recall'), html.Th('F1-Score'), html.Th('Support')
        ])

        rows = []
        for cls_name in class_names:
            if cls_name in report:
                m = report[cls_name]
                color = '#d4edda' if 'Healthy' in cls_name else '#f8d7da'
                rows.append(html.Tr([
                    html.Td(cls_name),
                    html.Td(f"{m['precision']:.3f}"),
                    html.Td(f"{m['recall']:.3f}"),
                    html.Td(f"{m['f1-score']:.3f}"),
                    html.Td(int(m['support'])),
                ], style={'backgroundColor': color}))

        table = html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={'width': '100%', 'borderCollapse': 'collapse',
                   'fontSize': '13px', 'textAlign': 'center'}
        )

        status = html.Div([
            html.Span(f"Test Accuracy: {accuracy:.2f}%",
                      style={'fontSize': '20px', 'fontWeight': 'bold'}),
            html.Span(f"  |  {len(all_labels)} samples evaluated",
                      style={'fontSize': '14px', 'color': 'gray'})
        ])

        return cm_fig, table, status
