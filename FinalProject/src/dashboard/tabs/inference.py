import base64
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

from src.utils.gradcam import GradCAM


def build_inference_tab():
    return html.Div([
        html.H3("Upload an Image for Detection & Classification"),

        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
            style={
                'width': '100%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                'textAlign': 'center', 'margin': '10px 0'
            },
            multiple=False
        ),

        html.Div(id='inference-results', children=[
            html.Div([
                html.Div([
                    html.H4("Original"),
                    html.Img(id='original-image', style={'maxWidth': '100%', 'maxHeight': '400px'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("Detection + Classification"),
                    html.Img(id='annotated-image', style={'maxWidth': '100%', 'maxHeight': '400px'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("GradCAM Attention"),
                    html.Img(id='gradcam-image', style={'maxWidth': '100%', 'maxHeight': '400px'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            html.Div([
                html.Div([
                    html.H4("Classification Result"),
                    html.Div(id='classification-result',
                             style={'fontSize': '24px', 'fontWeight': 'bold', 'padding': '10px'}),
                    html.Div(id='detection-confidence',
                             style={'fontSize': '16px', 'padding': '5px'}),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("Top-5 Class Probabilities"),
                    dcc.Graph(id='confidence-chart', style={'height': '300px'})
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),
        ], style={'display': 'none'})
    ])


def register_inference_callbacks(app, pipeline):

    @app.callback(
        [Output('original-image', 'src'),
         Output('annotated-image', 'src'),
         Output('gradcam-image', 'src'),
         Output('classification-result', 'children'),
         Output('detection-confidence', 'children'),
         Output('confidence-chart', 'figure'),
         Output('inference-results', 'style')],
        Input('upload-image', 'contents'),
        State('upload-image', 'filename'),
        prevent_initial_call=True
    )
    def process_upload(contents, filename):
        if contents is None:
            return [None] * 6 + [{'display': 'none'}]

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name

        img, detections = pipeline.detect(tmp_path)

        if img is None or not detections:
            return [
                contents, contents, None,
                "No object detected in image",
                "", go.Figure(),
                {'display': 'block'}
            ]

        best_det = max(detections, key=lambda d: d['det_confidence'])
        bbox = best_det['bbox']
        x1, y1, x2, y2 = bbox

        crop_bgr = img[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return [
                contents, contents, None,
                "Invalid detection region",
                "", go.Figure(),
                {'display': 'block'}
            ]

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        tensor = pipeline.transform(pil_crop).unsqueeze(0).to(pipeline.classifier.device)

        pipeline.classifier.model.eval()
        logits = pipeline.classifier.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_names = [pipeline.idx_to_class.get(int(idx), f"class_{idx}") for idx in top5_indices]
        top5_values = [float(p) for p in top5_probs]

        pred_name = top5_names[0]
        pred_prob = top5_values[0]

        annotated = pipeline._draw_annotation(img.copy(), {
            'bbox': bbox,
            'class_name': pred_name,
            'class_confidence': pred_prob
        })

        gradcam_overlay = _compute_gradcam(
            pipeline.classifier.model,
            tensor,
            crop_rgb
        )

        original_b64 = contents
        annotated_b64 = _numpy_to_b64(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        gradcam_b64 = _numpy_to_b64(gradcam_overlay) if gradcam_overlay is not None else None

        colors = ['green' if 'Healthy' in name else 'red' for name in top5_names]

        confidence_fig = go.Figure(data=[
            go.Bar(
                y=top5_names[::-1],
                x=top5_values[::-1],
                orientation='h',
                marker_color=colors[::-1],
                text=[f'{v:.2%}' for v in top5_values[::-1]],
                textposition='auto'
            )
        ])
        confidence_fig.update_layout(
            xaxis_title='Confidence',
            xaxis_range=[0, 1],
            margin=dict(l=10, r=10, t=10, b=30),
            height=250
        )

        condition_color = 'green' if 'Healthy' in pred_name else 'red'
        result_div = html.Span(
            f"{pred_name} ({pred_prob:.1%})",
            style={'color': condition_color}
        )

        det_info = f"Detection confidence: {best_det['det_confidence']:.2%}"

        Path(tmp_path).unlink(missing_ok=True)

        return [
            original_b64,
            f"data:image/png;base64,{annotated_b64}",
            f"data:image/png;base64,{gradcam_b64}" if gradcam_b64 else None,
            result_div,
            det_info,
            confidence_fig,
            {'display': 'block'}
        ]


def _compute_gradcam(model, input_tensor, original_rgb):
    try:
        target_layer = model.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        cam, pred_class = gradcam.generate(input_tensor)
        overlay = gradcam.overlay(original_rgb, cam)
        return overlay
    except Exception:
        return None


def _numpy_to_b64(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()
