from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL, no_update, ctx, dash_table
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.colors
import chardet
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, log_loss, confusion_matrix, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.svm import SVC
#from sklearn.manifold import TSNE
import umap
#from pacmap import PaCMAP
#from skopt import BayesSearchCV
import optuna
from optuna.samplers import TPESampler
from scipy.stats import norm, chi2_contingency, rankdata
from scipy.special import erf
#from scipy.spatial.distance import cdist
from multiprocessing import freeze_support
import io, base64, webbrowser, pickle, warnings, random, os
from itertools import product
from threading import Timer

#from sklearnex import patch_sklearn
#patch_sklearn()

warnings.filterwarnings("ignore")

app = Dash(external_stylesheets=[dbc.themes.SIMPLEX])
app.title = "ベイズ最適化ツール"

np.random.seed(0) # NumPyの乱数シードを固定
random.seed(0)    # Pythonの乱数シードを固定
sampler = TPESampler(
    seed=0,  # Optunaの乱数シードを固定
    multivariate=True,
    group=True,
    n_startup_trials=10,
    n_ei_candidates=32,
    consider_prior=True,
    prior_weight=1.0,
)

# レイアウト構築
app.layout = html.Div([
    dcc.Store(id="store-upload-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-original-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-original-categories", data={"data": [], "columns": []}),
    dcc.Store(id="store-num-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-class-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-label-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-rare-categories", data={"data": {}}),
    dcc.Tabs(id="tabs", value="upload-tab", children=[
        # データアップロードタブ
        dcc.Tab(
            id="upload-tab",
            value="upload-tab",
            label="①データアップロード",
            children=html.Div([
                html.Div("", style={"marginTop": "80px"}),  # タブ用のマージン
                dcc.Upload(
                    id="upload-data",
                    accept=".csv",
                    multiple=False,
                    children=html.Div([
                        html.P("CSVファイルをアップロード",
                               style={"fontSize": "22px", "fontWeight": "bold", "textAlign": "center", "marginTop": "10px"}),
                        html.P("クリックしてファイルを選択、またはここにファイルをドロップしてください",
                               style={"fontSize": "16px", "textAlign": "center", "marginTop": "10px"})
                    ]),
                ),
                html.Div(id="uploaded-data-info", style={"textAlign": "center", "padding": "50px"}),
                html.Div(
                    id="analysis-section",
                    style={"display": "none"},  # 初期状態は非表示
                    children=[
                        dbc.Accordion(
                            children=[
                                dbc.AccordionItem(
                                    title="アップロードデータ",
                                    children=[
                                        html.Div(id="data-table", style={"marginTop": "20px", "marginBottom": "20px"}),
                                        html.Div([
                                            html.Button("重複行を除外", id="exclude-duplicates", style={
                                                "margin": "5px", "padding": "5px 10px",
                                                "color": "#fff", "backgroundColor": "#e74c3c",
                                                "border": "none", "borderRadius": "5px",
                                                "cursor": "pointer"
                                            }),
                                            html.Button("表示されているデータをすべて除外", id="exclude-filtered", style={
                                                "margin": "5px", "padding": "5px 10px",
                                                "color": "#fff", "backgroundColor": "#f39c12",
                                                "border": "none", "borderRadius": "5px",
                                                "cursor": "pointer"
                                            }),
                                            html.Button("除外を全解除", id="reset-exclude", style={
                                                "margin": "5px", "padding": "5px 10px",
                                                "color": "#fff", "backgroundColor": "#3498db",
                                                "border": "none", "borderRadius": "5px",
                                                "cursor": "pointer"
                                            })
                                        ], style={"textAlign": "center", "marginBottom": "10px"}),
                                        
                                        html.H5("基礎統計量", style={"marginTop": "12px", "marginBottom": "8px"}),
                                        
                                        # 基本統計量テーブルを表示する場所
                                        html.Div(id="num-summary-table", style={"marginTop": "20px", "marginBottom": "10px"}),
                                        html.Div(id="cat-summary-table", style={"marginTop": "20px", "marginBottom": "20px"}),
                                    ]
                                )
                            ],
                            start_collapsed=False,
                            always_open=True,
                            style={"marginTop": "20px"}
                        ),
                        dbc.Accordion(
                            children=[
                                dbc.AccordionItem(
                                    title="データの相関",
                                    children=[
                                        # 上段：相関係数行列
                                        html.Div(
                                            id="corr-top",
                                            children=[
                                                dcc.Graph(
                                                    id="corr-matrix",
                                                    #config={"scrollZoom": True},
                                                    style={"display": "inline-block", "width": "50%", "height": "50vw", "vertical-align": "top"}
                                                ),
                                                dcc.Graph(
                                                    id="pair-plot",
                                                    #config={"scrollZoom": True},
                                                    style={"display": "inline-block", "width": "45%", "height": "45vw", "vertical-align": "top"}
                                                ),
                                            ]
                                        ),
                                        # 下段：相関比とクラメールのV
                                        html.Div(
                                            id="corr-bottom",
                                            children=[
                                                dcc.Graph(
                                                    id="eta-cramers-matrix",
                                                    style={"display": "inline-block", "width": "50%", "height": "50vw", "vertical-align": "top"}
                                                ),
                                                dcc.Graph(
                                                    id="crosstab",
                                                    style={"display": "inline-block", "width": "45%", "height": "45vw", "vertical-align": "top"}
                                                ),
                                            ],
                                            style={"marginTop": "30px"}  # 上段との間に余白
                                        )
                                    ]
                                )
                            ],
                            start_collapsed=False,
                            always_open=True,
                        ),
                        dbc.Accordion(
                            id="vif-accordion",
                            children=[
                                dbc.AccordionItem(title="多重共線性", children=[
                                        dcc.Graph(
                                            id="vif-matrix",
                                            #config={"scrollZoom": True},
                                            style={"display": "inline-block", "width": "70%", "height": "45vw"}
                                        )
                                ])
                            ],
                            start_collapsed=True,
                            always_open=True,
                            style={"marginBottom": "60px"}
                        )
                    ]
                )
            ], style={"textAlign": "center"})
        ),
        # 解析条件設定タブ
        dcc.Tab(
            id="condition-tab",
            value="condition-tab",
            label="②解析条件設定",
            style={"display": "none"},  # 初期状態は非表示
            children=[
                html.Div([
                    # 内部的なチェックボックス(常に非表示)
                    dcc.Checklist(id="explanatory-variables", options=[], value=[], style={"display": "none"}),
                    dcc.Checklist(id="target-variables", options=[], value=[], style={"display": "none"}),
                    dcc.Checklist(id="class-variables", options=[], value=[], style={"display": "none"}),
                    
                    html.Div("", style={"marginTop": "60px"}),  # タブ用のマージン
                    html.H3("解析条件を設定", style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px"}),
                    html.Div(
                        id="cat-summary-dropdown",
                        children=[
                            html.H5("エンコード設定", style={"marginTop": "12px", "marginBottom": "8px"}),
                            html.Div([
                                html.Label("レアカテゴリをまとめる:", style={"color": "#34495e", "marginLeft": "10px"}),
                                dbc.Switch(
                                    id="rare-category-switch",
                                    value=False,  # 初期状態では無効
                                    style={"marginLeft": "10px"}
                                )
                            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginTop": "20px"}),
                        ], style={"marginTop": "20px", "marginBottom": "20px"}
                    ),
                    html.Div([
                        html.Div([
                            html.Label("説明変数を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                            dcc.Checklist(
                                id="option-explanatory",
                                options=[],  # アップロード後に動的に更新
                                value=[],
                                style={"margin": "10px", "textAlign": "left"}
                            ),
                            html.Button("全て選択", id="select-all-explanatory", style={
                                "marginTop": "10px", "padding": "5px 10px",
                                "color": "#fff", "backgroundColor": "#00cc96",
                                "border": "none", "borderRadius": "5px",
                                "cursor": "pointer", "fontSize": "14px"
                            }),
                            html.Div(
                                id="impute-option",
                                children=[
                                    html.Label("欠損値補完:", style={
                                        "fontWeight": "bold",
                                        "color": "#34495e",
                                        "marginBottom": "5px",
                                        "marginTop": "10px"
                                    }),
                                    dcc.Dropdown(
                                        id="impute-mode",
                                        options=[
                                            {"label": "補完なし", "value": "none"},
                                            {"label": "中央値/最頻値", "value": "median_mode"},
                                            {"label": "線形補間/「欠損」埋め", "value": "iterative_missing"},
                                        ],
                                        value="none",
                                        clearable=False,
                                        style={"flex": "1", "maxWidth": "250px", "paddingLeft": "10px"}
                                    )
                                ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}
                            )
                        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "15px",
                                  "borderRadius": "10px", "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"}),
                        html.Div([
                            html.Label("目的変数(回帰)を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                            dcc.Checklist(
                                id="option-target",
                                options=[],  # アップロード後に動的に更新
                                value=[],
                                style={"margin": "10px", "textAlign": "left"}
                            ),
                            html.Button("全て選択", id="select-all-target", style={
                                "marginTop": "10px", "padding": "5px 10px",
                                "color": "#fff", "backgroundColor": "#00cc96",
                                "border": "none", "borderRadius": "5px",
                                "cursor": "pointer", "fontSize": "14px"
                            })
                        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "15px",
                                  "borderRadius": "10px", "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"}),
                        html.Div([
                            html.Div([
                                html.Label("目的変数(分類)を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                                dcc.Checklist(
                                    id="option-class",
                                    options=[],  # アップロード後に動的に更新
                                    value=[],
                                    style={"margin": "10px", "textAlign": "left"}
                                ),
                                html.Button("全て選択", id="select-all-class", style={
                                "marginTop": "10px", "padding": "5px 10px",
                                "color": "#fff", "backgroundColor": "#00cc96",
                                "border": "none", "borderRadius": "5px",
                                "cursor": "pointer", "fontSize": "14px"
                                })
                            ],style={"width": "100%", "padding": "15px", "borderRadius": "10px",
                                      "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)", "marginBottom": "15px",
                                      "display": "block"}),
                            html.Div([
                                html.Label("ラベル変数を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                                dcc.RadioItems(
                                    id="label-variables",
                                    options=[],  # アップロード後に動的に更新
                                    value=None,
                                    style={"margin": "10px", "textAlign": "left"}
                                ),
                                html.Div([
                                    html.Label("散布図の色分け:", style={"color": "#34495e", "marginLeft": "10px"}),
                                    dbc.Switch(
                                        id="color-toggle",
                                        value=False,  # 初期状態では無効
                                        style={"marginLeft": "10px"}
                                    )
                                ], style={"display": "flex", "alignItems": "center", "marginTop": "20px"})
                            ], style={"width": "100%", "padding": "15px", "borderRadius": "10px",
                              "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"})
                        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top",
                          "borderRadius": "10px", "backgroundColor": "#f9fbfc"})
                    ], style={"width": "90%", "margin": "0 auto", "display": "flex", "justifyContent": "space-between"}),
                    html.Div([
                        dbc.Accordion([
                            dbc.AccordionItem([
                                html.Div([
                                    # --- 評価指標選択 ---
                                    html.Div([
                                        html.Label("評価指標:", style={"fontWeight": "bold"}),
                                        dcc.RadioItems(
                                            id="tuning-type",
                                            options=[
                                                {"label": "安定： 外れ値に強い指標で最適化します（Huber/Brier）", "value": "stable"},
                                                {"label": "精密： 予測の正確さを厳密に評価します（RMSE/Logloss）", "value": "precise"},
                                            ],
                                            value="stable", # 初期値
                                            labelStyle={"display": "block", "marginBottom": "5px", "fontSize": "14px"}
                                        ),
                                    ], style={"width": "48%", "display": "inline-block", "textAlign": "left", "verticalAlign": "top"}),
                                ], style={"padding": "10px"})
                            ], title="詳細設定")
                        ], start_collapsed=True, style={"marginTop": "20px"})
                    ], style={"width": "90%", "margin": "0 auto"}),
                    html.Button("解析開始", id="analyze-button", n_clicks=0, style={
                        "padding": "10px 20px", "marginTop": "20px",
                        "color": "#fff", "backgroundColor": "#3498db",
                        "border": "none", "borderRadius": "5px",
                        "cursor": "pointer", "fontSize": "16px"
                    })
                ], style={"width": "80%", "margin": "0 auto", "textAlign": "center", "padding": "20px", "backgroundColor": "#f9fbfc",
                          "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
            ]
        ),
        # 解析結果タブ
        dcc.Tab(
            id="result-tab",
            value="result-tab",
            label="③解析結果",
            style={"display": "none"},
            children=html.Div([
                dcc.Tabs(
                    id="dynamic-tabs",
                    children=[],
                    style={
                        "position": "fixed", 
                        "top": "65px", 
                        "left": "50%", 
                        "width": "95%", 
                        "transform": "translateX(-50%)", 
                        "zIndex": 1000,
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                    }
            )],
            style={"textAlign": "center", "width": "90%", "margin": "0 auto", "padding": "20px",
                      "backgroundColor": "#f9f9f9", "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
        ),
        # 予測値計算タブ
        dcc.Tab(
            id="prediction-tab",
            value="prediction-tab",
            label="④予測値計算",
            style={"display": "none"},  # 初期状態は非表示
            children=[
                dcc.Store(id="store-scalers", data={}),
                html.Div("", style={"marginTop": "60px"}),  # タブ用のマージン
                html.Div([
                    # 左側のスクロール可能なスライダー部分
                    html.Div(
                        id="slider-section",
                        style={
                            "width": "35%",  # スライダー部分の幅
                            "height": "calc(100vh - 60px)",  # 画面全体の高さ
                            "overflowY": "scroll",  # 縦方向スクロールを有効化
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "20px",
                            "boxSizing": "border-box"
                        },
                        children=[
                            html.Div("", style={"marginTop": "60px"}),  # タブ用のマージン
                            html.Div(id="input-fields")  # スライダーなどを動的に配置
                        ]
                    ),
                    # 右側の予測結果部分
                    html.Div(
                        id="prediction-results",
                        style={
                            "width": "60%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "20px",
                            "borderLeft": "1px solid #ddd",
                            "position": "fixed",
                            "right": "2%",
                            "top": "50%",
                            "transform": "translateY(-50%)",
                            "boxSizing": "border-box",
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
                        }
                    )
                ], style={"display": "flex", "justifyContent": "space-between", "width": "90%", "margin": "0 auto"})
            ]
        ),
        # 推奨実験条件タブ
        dcc.Tab(
            id="recommend-tab",
            value="recommend-tab",
            label="⑤推奨実験条件",
            children=dbc.Container([
                html.Div("", style={"marginTop": "80px"}),  # タブ用のマージン
                dbc.Row([
                    dbc.Col([dbc.Label("実験点の数:", style={"textAlign": "left"}),], width=1),
                    dbc.Col([dbc.Input(id='num-exp', type='number', value=10, min=1, max=20),], width=2),
                ], align="center"),  # 縦方向に中央揃え
                dbc.Row([
                    dbc.Col([dbc.Label("モード:", style={"textAlign": "left"}),], width=1),
                    dbc.Col([
                        dcc.Dropdown(
                            id='opt-type',
                            options=[
                                {'label': '探索重視', 'value': '4'},
                                {'label': 'バランス', 'value': '2'},
                                {'label': '活用重視', 'value': '1'},
                            ],
                            value='2',  # デフォルト値をバランスに設定
                            placeholder="基準を選択"
                        ),
                    ], width=2),  # ドロップダウンの幅
                    dbc.Col([html.Div(id='description-output')], width=8)
                ], align="center"),  # 縦方向に中央揃え
                html.Hr(),
                html.H4("実験点の設定", style={"marginTop": "20px"}),
                # 回帰セクション
                html.Div(id="target-variable-table", children=[], style={"marginTop": "20px"}),
                # 分類セクション
                html.Div(id="class-variable-table", children=[], style={"marginTop": "20px"}),
                # 説明変数
                html.H5("探索範囲の設定", style={"color": "#5C7B99", "fontWeight": "bold", "margin-top": "30px"}),
                dag.AgGrid(
                    id="explanatory-variable-table",
                    columnDefs=[
                        {
                            "headerName": "説明変数",
                            "field": "説明変数",
                            "editable": False,
                            "flex": 1
                        },
                        {
                            "headerName": "水準値(数値をカンマ区切りで入力)",
                            "field": "水準値",
                            "editable": True,
                            "flex": 1,
                            "cellStyle": {
                                "cursor": "text",
                                "border": "1px solid #999",
                            }
                        }
                    ],
                    rowData=[],
                    defaultColDef={"resizable": True, "filter": False,"sortable": False},
                    dashGridOptions={"domLayout": "autoHeight", "rowSelection": "single", "singleClickEdit": True}
                ),
                # 制約条件設定
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Div(id='constraints-container', children=[]),
                        dash_table.DataTable(
                            id="constraints-table",
                            columns=[
                                {"name": "変数", "id": "変数"},
                                {"name": "合計値制約", "id": "合計値制約"},
                                {"name": "選択数制約", "id": "選択数制約"},
                            ],
                            data=[],
                            row_deletable=True,
                            style_table={"margin-bottom": "20px"},
                            style_cell={"textAlign": "center", "padding": "5px", "fontFamily": "Yu Gothic, Meiryo, sans-serif"},
                            style_header={"fontWeight": "bold"}
                        ),
                        dbc.Button("制約条件を追加", id="add-constraint-btn"),
                    ], title="制約条件設定", style={"marginTop": "10px"}),
                ], start_collapsed=True),
                
                dbc.Col([
                    dbc.Button("実験候補点生成", id="recommend-button", color="success", className="mt-3"),
                    dcc.Loading(
                        id="loading",
                        type="circle",
                        children=html.Div(id="output-results", children=[""], style={"margin-top": "20px"})
                    ),
                    html.Div(id="prediction-container"),  # ヒートマップを表示するコンテナ
                    html.Div(id="map-container")  # 分布マップを表示するコンテナ
                ], width=12)
            ]),
            style={"display": "none", "textAlign": "center", "width": "90%", "margin": "0 auto", "padding": "20px",
                      "backgroundColor": "#f9f9f9", "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}
        ),  
    ],
    style={
        "position": "fixed",
        "top": "0",
        "width": "100%",
        "zIndex": 999,
        "backgroundColor": "white",
        "borderBottom": "1px solid #ddd",
        "textOverflow": "ellipsis",
        "whiteSpace": "nowrap"
    }),
    dbc.Modal(  #制約条件のモーダル
        [
            dbc.ModalHeader("制約条件を入力"),
            dbc.ModalBody([
                dbc.Checklist(
                    id="variable-checkboxes",
                    options=[],
                    value=[],
                    inline=True
                ),
                dbc.Row([
                    dbc.Col(dbc.Label("合計値制約（最小～最大）:", style={"textAlign": "left"}), width=6),
                    dbc.Col(dbc.Input(id="sum-constraint-min", type='number', placeholder='最小値'), width=3),
                    dbc.Col(dbc.Input(id="sum-constraint-max", type='number', placeholder='最大値'), width=3),
                ], className="mt-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("選択数制約（最小～最大）:", style={"textAlign": "left"}), width=6),
                    dbc.Col(dbc.Input(id="select-count-constraint-min", type='number', placeholder='最小値'), width=3),
                    dbc.Col(dbc.Input(id="select-count-constraint-max", type='number', placeholder='最大値'), width=3),
                ], className="mt-2"),
                html.Div(id="error-message", style={"color": "red", "margin-top": "10px", "textAlign": "left"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("保存", id="save-constraint-btn", color="success", disabled=False),
                dbc.Button("閉じる", id="close-modal-btn", color="secondary"),
            ]),
        ],
        id="constraint-modal",
        is_open=False,
    ),
])
# コールバック: 目的変数ごとに動的にタブを生成
@app.callback(
    Output("dynamic-tabs", "children"),
    Input("analyze-button", "n_clicks"),
    [State("store-num-data", "data"),
     State("target-variables", "value"),
     State("class-variables", "value")]
)
def update_dynamic_tabs(n_clicks, stored_data, target_variables, class_variables):
    if not stored_data or (not target_variables and not class_variables):
        return [dcc.Tab(label="No Data", children=[html.Div("目的変数が選択されていません。")])]

    tabs = []
    total_variables = target_variables + class_variables
    for target in total_variables:
        children_elements=[
            dcc.Store(id={"type": "saved-cv-scores", "index": target}),
            dcc.Store(id={"type": "stored-model", "index": target}),
            html.Div("", style={"marginTop": "130px"}),
            html.Div(
                [
                    html.Div(
                        id={"type": "result-message", "index": target},
                        style={
                            "padding": "10px 20px",
                            "color": "#2c3e50",
                            "fontSize": "16px",
                            "display": "inline-block",
                            "marginTop": "15px",
                            "marginRight": "20px"
                        }
                    ),
                    html.Div(
                        [
                            html.Span("CV結果: ", style={"marginRight": "10px"}),
                            dash_table.DataTable(
                                id={"type": "cv-export-table", "index": target},
                                columns=[
                                    {"name": "行数", "id": "行数"},
                                    {"name": "fold", "id": "fold"},
                                    {"name": "y_true", "id": "y_true"},
                                    {"name": "y_pred", "id": "y_pred"},
                                ],
                                data=[],
                                export_format="csv",
                                page_action="none",
                                style_table={
                                    "height": "0px",
                                    "width": "0px",
                                    "overflow": "hidden",
                                    "border": "none",
                                },
                                style_cell={"padding": "0", "height": "0px"},
                            )
                        ],
                        id={"type": "cv-export-container", "index": target},
                        style={"display": "none", "alignItems": "center", "gap": "10px", "padding": "10px 0"}
                    )
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "padding": "10px 20px"
                }
            ),
            dcc.Graph(
                id={"type": "feature-importance-plot", "index": target},
                figure=default_fig(title="", center_text="解析中..."),
                #config={"scrollZoom": True},
                style={"display": "inline-block", "width": "90%", "height": "40vw"}
            )

        ]
        # 分類と回帰でレイアウトを切り替え
        if target in class_variables:
            # 分類の場合
            children_elements.append(
                dcc.Graph(
                    id={"type": "scatter-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    style={"display": "inline-block", "width": "50%", "height": "50vw"}
                )
            )
            children_elements.append(
                html.Div([
                    dcc.Slider(
                        id={"type": "threshold-slider", "index": target},
                        min=0.0,
                        max=1.0,
                        value=0.5,
                        marks={i: f"{i:.1f}" for i in [round(x * 0.1, 1) for x in range(0, 11)]},
                        tooltip={"always_visible": True},
                        vertical=True  # 縦型スライダーを指定
                    ),
                    html.Label("閾値", style={"fontWeight": "bold"}),
                    html.Label("", style={"fontWeight": "bold", "marginBottom": "10vw"})
                ], style={"display": "inline-block", "width": "5%", "height": "30vw"}),
            )
            children_elements.append(
                dcc.Graph(
                    id={"type": "cross-validation-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "50vw"}
                )
            )
        else:
            # 回帰の場合
            children_elements.append(
                dcc.Graph(
                    id={"type": "scatter-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "50vw"}
                )
            )
            children_elements.append(
                html.Div([
                    dcc.Slider(id={"type": "threshold-slider", "index": target},min=0, max=1, value=0),
                    ], style={"display": "none"})
            )
            children_elements.append(
                dcc.Graph(
                    id={"type": "cross-validation-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "50vw"}
                )
            )
            
        tabs.append(
            dcc.Tab(id={"type": "child-tabs", "index": target}, label=f"{target}", children=
                dcc.Loading(
                    type="circle",  # プログレスリングを表示
                    id={"type": "loading", "index": target},
                    children=children_elements
                )
            )
        )

    return tabs

# Exportボタン表示制御
@app.callback(
    Output({"type": "cv-export-container", "index": ALL}, "style"),
    Input("analyze-button", "n_clicks"),
    Input({"type": "cv-export-table", "index": ALL}, "data"),
    prevent_initial_call=True
)
def toggle_export_container(n_clicks, all_data):
    styles = []
    for data in all_data:
        if data and len(data) > 0:
            styles.append({"display": "flex"})
        else:
            styles.append({"display": "none"})
    return styles

# 各モードの説明文
descriptions = {
    "4": "未知の領域を試し、良い結果が出る条件を探ります。実験序盤、もしくは行き詰ったときに適しています。",
    "2": "未知の領域と既知の良い条件をバランスよく探索します。実験序盤～中盤に適しています。",
    "1": "既に良い結果が得られている条件を深堀りし、最適な結果を目指します。実験終盤に適しています。",
}
# コールバックで説明文を更新
@app.callback(
    Output("description-output", "children"),
    Input("opt-type", "value")
)
def update_description(selected_value):
    if selected_value:
        return descriptions[selected_value]
    return "選択したモードの説明がここに表示されます。"

# チェックボックス生成
@app.callback(
    Output("variable-checkboxes", "options"),
    Input("explanatory-variable-table", 'rowData'),
    prevent_initial_call=True
)
def update_checkboxes(variable_table_data):
    options = []

    if variable_table_data:
        options = [
            {"label": row["説明変数"], "value": row["説明変数"]}
            for row in variable_table_data if row["説明変数"]
        ]
        return options
    else:
        return []  # データがない場合は空リストを返す

# モーダルの制御
@app.callback(
    [Output("error-message", "children"),
     Output("save-constraint-btn", "disabled")],
    [Input("variable-checkboxes", "options"),
     Input("variable-checkboxes", "value"),
     Input("sum-constraint-min", "value"),
     Input("sum-constraint-max", "value"),
     Input("select-count-constraint-min", "value"),
     Input("select-count-constraint-max", "value")]
)
def validate_constraints(options, checked_vars, sum_min, sum_max, select_min, select_max):
    max_allowed = len(checked_vars)  # 選択された変数の数
    errors = []

    # options が空かどうか確認**
    if not options or len(options) == 0:
        errors.append("説明変数が選択されていません。")
        return " ".join(errors), True

    # チェックされた変数がない場合
    if not checked_vars:
        errors.append("少なくとも1つの変数を選択してください。")
    else:
        # 選択数制約の範囲チェック
        if select_min is not None and (select_min < 0 or select_min > max_allowed):
            errors.append(f"選択数制約（最小値）は0以上、{max_allowed}以下で入力してください。")
        if select_max is not None and (select_max < 0 or select_max > max_allowed):
            errors.append(f"選択数制約（最大値）は0以上、{max_allowed}以下で入力してください。")
        if select_min is not None and select_max is not None and select_min > select_max:
            errors.append("最小値は最大値以下で入力してください。")
        else:
            # 合計値制約の範囲チェック
            if sum_min is not None and sum_max is not None and sum_min > sum_max:
                errors.append("最小値は最大値以下で入力してください。")

    # エラーがある場合、保存ボタンを無効化
    error_message = " ".join(errors)
    return error_message, bool(errors)

@app.callback(
    [Output("constraint-modal", "is_open"),
     Output("constraints-table", "data"),
     Output("variable-checkboxes", "value"),
     Output("sum-constraint-min", "value"),
     Output("sum-constraint-max", "value"),
     Output("select-count-constraint-min", "value"),
     Output("select-count-constraint-max", "value")],
    [Input("save-constraint-btn", "n_clicks"),
     Input("add-constraint-btn", "n_clicks"),
     Input("close-modal-btn", "n_clicks")],
    [State("variable-checkboxes", "value"),
     State("sum-constraint-min", "value"),
     State("sum-constraint-max", "value"),
     State("select-count-constraint-min", "value"),
     State("select-count-constraint-max", "value"),
     State("constraints-table", "data"),
     State("constraint-modal", "is_open")],
    prevent_initial_call=True
)
def handle_modal_and_save(save_clicks, open_clicks, close_clicks,
                          selected_vars, sum_min, sum_max, select_min, select_max,
                          current_data, is_open):
    ctx_trigger = ctx.triggered_id  # 現在のトリガーを取得

    # 保存ボタンが押された場合（保存処理）
    if ctx_trigger == "save-constraint-btn":
        if sum_min is None : sum_min = ""
        if sum_max is None : sum_max = ""
        if select_min is None : select_min = 0
        if select_max is None : select_max = len(selected_vars)
        new_row = {
            "変数": ", ".join(selected_vars) if selected_vars else "なし",
            "合計値制約": f"{sum_min}～{sum_max}" if sum_min != "" or sum_max != "" else "なし",
            "選択数制約": f"{select_min}～{select_max}"
        }
        current_data.append(new_row)
        # フィールドをリセットしてモーダルを閉じる
        return False, current_data, [], None, None, None, None

    # モーダルの開閉トグル（開く/閉じるボタン）
    if ctx_trigger in ["add-constraint-btn", "close-modal-btn"]:
        return not is_open, current_data, selected_vars, sum_min, sum_max, select_min, select_max

    # デフォルト（何も変更しない）
    return is_open, current_data, selected_vars, sum_min, sum_max, select_min, select_max

def convert_percent_columns(df):
    """
    データフレーム内の '10%' や '20.5％' などのパーセント表記文字列を検出し、
    自動的に数値（例: 0.1, 0.205）に変換して float 型の列にする関数。
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            def try_parse_pct(val):
                if pd.isna(val) or val is None or val == "":
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    s = val.strip()
                    if s.endswith("%") or s.endswith("％"):
                        num_part = s[:-1].strip().replace(",", "")
                        try:
                            return float(num_part) / 100.0
                        except ValueError:
                            return val
                    else:
                        try:
                            return float(s.replace(",", ""))
                        except ValueError:
                            return val
                return val

            has_pct = df[col].astype(str).str.contains(r"[%％]", na=False)
            if has_pct.any():
                converted_series = df[col].map(try_parse_pct)
                converted_numeric = pd.to_numeric(converted_series, errors="coerce")
                valid_orig = df[col].notna() & (df[col].astype(str).str.strip() != "")
                if valid_orig.sum() > 0 and converted_numeric[valid_orig].notna().all():
                    df[col] = converted_numeric
    return df

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, impute_method="none", scale=True):
        self.impute_method = impute_method
        self.scale = scale

    def fit(self, X, y=None):
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        # --- 補完器を作成 ---
        if self.impute_method in ["median", "median_mode"]:
            self.imputer = SimpleImputer(strategy="median")
        elif self.impute_method in ["linear", "iterative_missing"]:
            self.imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
        elif self.impute_method == "zero":
            self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        elif self.impute_method == "non-linear":
            self.imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(
                    n_estimators=30, max_depth=5, random_state=0, n_jobs=-1
                ),
                max_iter=2,
                initial_strategy="median",
                random_state=0
            )
        else:
            # 何もしない場合
            self.imputer = None

        # --- fit ---
        if self.imputer is not None:
            X_imputed = self.imputer.fit_transform(X_values)
        else:
            X_imputed = X_values.astype(float)

        # --- スケーラー fit ---
        if self.scale:
            self.scaler = RobustScaler()
            self.scaler.fit(X_imputed)
        else:
            return X_imputed

        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        index = X.index if is_df else None
        X_values = X.values if is_df else X

        # --- transform ---
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X_values)
        else:
            X_imputed = X_values.astype(float)

        if self.scale:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed

        if is_df:
            return pd.DataFrame(X_scaled, columns=columns, index=index)
        return X_scaled


def default_fig(title="", center_text=""):
    fig = go.Figure()
    fig.update_layout(
        title={"text": title},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        dragmode=False,
        shapes=[{"type": "rect", "xref": "paper", "yref": "paper", "x0": 0, "y0": 0, "x1": 1, "y1": 1, "line": {"color": "lightgray", "dash": "dash"}}],
        annotations=[{"text": center_text, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 16, "color": "black"}, "align": "center"}]
    )
    return fig

def generate_fold_colors(n_folds):
    color_scale = plotly.colors.sequential.Viridis  # Viridisカラースケールを使用
    colors = [color_scale[i * len(color_scale) // n_folds] for i in range(n_folds)]
    return colors

def scatter_layout(fig, x, y, title):
    x = np.array(x)
    y = np.array(y)

    # 回帰直線データを計算
    regression_model = LinearRegression()
    regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    y_line = regression_model.predict(x.reshape(-1, 1))
    r2 = concordance_corrcoef(x, y)  #r2_score(x, y)

    # 散布図
    fig.add_trace(go.Scatter(x=x, y=y_line.flatten(), mode="lines", name="回帰直線", line=dict(color="red")))
    padding = (max(x) - min(x)) * 0.05  # 範囲の5%を余白として追加
    min_val = min(x.min(), y.min()) - padding
    max_val = max(x.max(), y.max()) + padding

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="対角線",
        line=dict(color="rgba(128, 128, 128, 0.5)", dash="dash")
    ))

    fig.update_layout(
        title=f"{title} [一致度: {r2:.4f}]",
        xaxis=dict(title="実測値", range=[min_val, max_val], scaleanchor="y", constrain="domain"),
        yaxis=dict(title="予測値", range=[min_val, max_val], constrain="domain"),
        showlegend=True,
        #dragmode="pan"
    )
    return fig

def load_model(model_data, variable_name):
    if not model_data:
        return None
    for entry in model_data:
        if isinstance(entry, dict) and entry.get("name") == variable_name:
            try:
                return pickle.loads(base64.b64decode(entry["model"].encode('utf-8')))
            except Exception as e:
                print(f"[load_model] モデルのデコード失敗: {e}")
                return None
    return None

def concordance_corrcoef(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def huber_loss(y_true, y_pred, delta=1.345):
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    squared = 0.5 * error**2
    linear = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small, squared, linear).mean()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Huber: 低い方が良い
huber_scorer = make_scorer(huber_loss, greater_is_better=False)
# RMSE: 低い方が良い
rmse_scorer = make_scorer(rmse, greater_is_better=False)

def brier_score(y_true, y_prob): 
    if y_prob.ndim == 1: 
        # binary 
        return np.mean((y_prob - y_true)**2) 
    else: 
        # multi-class 
        classes = np.unique(y_true) # クラス → インデックスの辞書を作る
        class_to_index = {cl: i for i, cl in enumerate(classes)} # y_true を整数インデックスに変換 
        y_int = np.array([class_to_index[v] for v in y_true]) # one-hot 生成 
        y_true_onehot = np.eye(y_prob.shape[1])[y_int] 
        return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    
def logloss(y_true, y_prob):
    return log_loss(y_true, y_prob)

# コールバック: タブの表示管理
@app.callback(
    [Output("condition-tab", "style"),
     Output("result-tab", "style"),
     Output("prediction-tab", "style"),
     Output("recommend-tab", "style"),
     Output("tabs", "value")],
    [Input("upload-data", "contents"),
     Input("dynamic-tabs", "children")],
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def manage_tabs(upload, dinamic_tab, contents):
    # トリガーされたイベントを判定
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered_id == "upload-data":
        # データアップロード時に結果タブを非表示
        return {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, "upload-tab"

    if triggered_id == "dynamic-tabs" and contents:
        # 解析ボタン押下時に結果タブを表示
        return {"display": "block"}, {"display": "block"}, {"display": "block"}, {"display": "block"}, "result-tab"

    # 初期状態（どのトリガーも発生していない場合）
    return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, "upload-tab"


# コールバック: ファイルアップロード時の処理
@app.callback(
    [Output("upload-data", "style"),
     Output("uploaded-data-info", "children"),
     Output("store-upload-data", "data"),
     Output("data-table", "children"),
     Output("analysis-section", "style")],
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def process_upload(contents, filename):
    upload_style = {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "flexDirection": "column",
        "height": "150px",
        "width": "50%",
        "margin": "0 auto",
        "border": "2px dashed #cccccc",
        "borderRadius": "10px",
        "backgroundColor": "#f9f9f9",
        "textAlign": "center",
        "cursor": "pointer"
    }

    # --- ファイル未アップロード時 ---
    if not contents or not filename:
        return (
            upload_style,
            "ファイルがアップロードされていません。",
            None,
            html.Div("", style={"textAlign": "center", "color": "#888", "padding": "20px"}),
            {"display": "none"}
        )

    # --- ファイルアップロード時 ---
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # エンコード形式を判定
        result = chardet.detect(decoded)
        encoding = result["encoding"] if result["encoding"] else "utf-8"

        # データを読み込む
        try:
            data = pd.read_csv(io.StringIO(decoded.decode(encoding)))
        except:
            data = pd.read_csv(io.StringIO(decoded.decode("cp932", errors="replace")))

        # 不要なUnnamed列を除去
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

        # 「%」付き表記を数値に変換（例: 10% -> 0.1）
        data = convert_percent_columns(data)

        # NumPy 型 → Python ネイティブ型へ変換
        data = data.where(pd.notnull(data), None)
        data = data.astype(object)
        data = data.map(lambda x: x.item() if hasattr(x, "item") else x)

        # 基本情報
        n_rows, n_cols = data.shape
        n_num = data.select_dtypes(include=[np.number]).shape[1]
        n_cat = data.select_dtypes(exclude=[np.number]).shape[1]
        info_message = [
            f"アップロードされたファイル: {filename}", html.Br(),
            f"{n_rows}行×{n_cols}列  (数値列: {n_num}, カテゴリ列: {n_cat})"
        ]

        # 保存用データ（元列名のまま）
        save_data = data.copy()
        save_data.insert(0, "行番号", range(1, len(save_data) + 1))
        save_data["除外"] = False

        # 表示用データ（列名の . を ． に変換）
        grid_data = save_data.copy()
        grid_data.columns = [
            c.replace(".", "．") if isinstance(c, str) else c
            for c in grid_data.columns
        ]

        table_data = grid_data.to_dict("records")

        # columnDefs（表示用の列名に合わせる）
        column_defs = [
            {
                "headerName": "除外",
                "field": "除外",
                "cellEditor": "agCheckboxCellEditor",
                "editable": True,
                "pinned": "left",
                "width": 70,
                "cellStyle": {"textAlign": "center"}
            },
            {
                "headerName": "行番号",
                "field": "行番号",
                "sortable": True,
                "filter": True,
                "pinned": "left",
                "cellStyle": {
                    "backgroundColor": "#f0f0f0",
                    "fontWeight": "bold",
                    "textAlign": "right"
                },
                "width": 70
            }
        ] + [
            {
                "headerName": col.replace(".", "．") if isinstance(col, str) else col,
                "field": col.replace(".", "．") if isinstance(col, str) else col,
                "sortable": True,
                "filter": True,
                "width": 45 + len(str(col)) * 12
            }
            for col in data.columns
        ]

        data_table = dag.AgGrid(
            id="ag-grid",
            columnDefs=column_defs,
            rowData=table_data,
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions={
                "suppressRowClickSelection": True,
                "undoRedoCellEditing": True,
                "singleClickEdit": True,
                "stopEditingWhenCellsLoseFocus": True,
            },
            style={
                "height": "500px",
                "width": "95%",
                "margin": "0 auto",
                "border": "1px solid #ddd"
            },
        )

        return (
            {"display": "none"},
            info_message,
            {"data": save_data.to_dict("records"), "columns": list(save_data.columns)},
            data_table,
            {"display": "block"}
        )

    except Exception as e:
        return (
            upload_style,
            f"エラー: {str(e)}",
            None,
            html.Div(
                "エラーが発生しました。データが処理できません。",
                style={"textAlign": "center", "color": "red", "padding": "20px"}
            ),
            {"display": "none"}
        )

@app.callback(
    Output("ag-grid", "rowData", allow_duplicate=True),
    [Input("exclude-duplicates", "n_clicks"),
     Input("exclude-filtered", "n_clicks"),
     Input("reset-exclude", "n_clicks")],
    [State("ag-grid", "rowData"),
     State("ag-grid", "virtualRowData")],
    prevent_initial_call=True
)
def update_exclude(dup_clicks, filter_clicks, reset_clicks, row_data, virtual_data):
    if not ctx.triggered or row_data is None:
        return no_update

    triggered_id = ctx.triggered_id
    df = pd.DataFrame(row_data)

    if "除外" not in df.columns:
        df["除外"] = False

    if triggered_id == "exclude-duplicates":
        dup_mask = df.duplicated(
            subset=df.columns.difference(["除外", "行番号"]),
            keep="last"
        )
        df.loc[dup_mask, "除外"] = True

    elif triggered_id == "exclude-filtered" and virtual_data:
        v_indices = pd.DataFrame(virtual_data)["行番号"].values
        df.loc[df["行番号"].isin(v_indices), "除外"] = True

    elif triggered_id == "reset-exclude":
        df["除外"] = False

    return df.to_dict("records")


@app.callback(
    Output("store-original-data", "data"),
    [Input("store-upload-data", "data"),
     Input("ag-grid", "rowData"),
     Input("ag-grid", "cellValueChanged")],
    prevent_initial_call=True
)
def update_store(upload_data, rowData, cellChanged):
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return no_update

    # アップロード直後
    if triggered_id == "store-upload-data" and upload_data is not None:
        df = pd.DataFrame(upload_data["data"], columns=upload_data["columns"])

    # チェックボックス変更時
    elif triggered_id == "ag-grid" and cellChanged:
        df = pd.DataFrame(rowData)
        changed_row = cellChanged[0]["data"]
        changed_id = changed_row["行番号"]
        exclude_flag = changed_row["除外"]
        df.loc[df["行番号"] == changed_id, "除外"] = exclude_flag

    # rowData 更新時（ボタン操作など）
    elif triggered_id == "ag-grid" and rowData is not None:
        df = pd.DataFrame(rowData)

    else:
        return no_update

    if "除外" in df.columns:
        df_filtered = df[df["除外"] == False].drop(columns=["除外"], errors="ignore")
    else:
        df_filtered = df

    # 表示用の全角ドットを元に戻して保存
    df_filtered.columns = [
        c.replace("．", ".") if isinstance(c, str) else c
        for c in df_filtered.columns
    ]

    return {
        "data": df_filtered.to_dict("records"),
        "columns": list(df_filtered.columns)
    }

@app.callback(
    [Output("num-summary-table", "children"),
     Output("cat-summary-table", "children"),
     Output("cat-summary-dropdown", "children")],
    Input("store-original-data", "data")
)
def update_summary_table(stored_data):
    if stored_data is None:
        return (
            "データがありません",
            "データがありません",
            html.Div([
                dcc.Dropdown(
                    id={"type": "dynamic-dropdown", "col": "__dummy__"},
                    options=[{"label": "しない", "value": "しない"}],
                    value="しない",
                    style={"display": "none"}
                )
            ])
        )

    df = pd.DataFrame(stored_data["data"], columns=stored_data["columns"])
    # 数値データのみ抽出
    df_numeric = df.select_dtypes(include=[np.number])
    df_category = df.select_dtypes(exclude=[np.number])

    # --- 数値データのサマリー ---
    def summary_stats(df):
        stats = []
        n_total = len(df)
        for col in df.columns:
            if col == "行番号":
                continue
            n = df[col].count()
            mean = df[col].mean()
            median = df[col].median()
            max_val = df[col].max()
            min_val = df[col].min()
            n_missing = df[col].isna().sum()
            missing_pct = (n_missing / n_total) * 100
            std = df[col].std(ddof=1)
    
            stats.append({
                "変数": col,
                "n": n,
                "平均値": round(mean, 3),
                "標準偏差": round(std, 3),
                "最小値": round(min_val, 3),
                "中央値": round(median, 3),
                "最大値": round(max_val, 3),
                "欠損数": f"{n_missing} ({missing_pct:.1f}%)" 
            })
        return pd.DataFrame(stats)
    
    # 基本統計量を計算
    if df_numeric.shape[1] > 0:
        summary_df = summary_stats(df_numeric)
        num_summary_table = dash_table.DataTable(
            data=summary_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in summary_df.columns],
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": "bold"},
            style_cell={"textAlign": "center", "fontFamily": "Yu Gothic, Meiryo, sans-serif"}
        )
    else:
        num_summary_table = ""  # 数値データがないときは表示しない

    # --- カテゴリデータのサマリー ---
    def cat_summary_stats(df):
        stats = []
        n_total = len(df)
        for col in df.columns:
            n = df[col].count()
            n_missing = df[col].isna().sum()
            missing_pct = (n_missing / n_total) * 100
            n_unique = df[col].nunique(dropna=True)
            unique_ratio = (n_unique / n) *100
            value_counts = df[col].value_counts(dropna=True)  # 出現回数のカウント 
            rare_threshold = int(np.ceil(np.sqrt(n_total) / 5))
            rare_rows = value_counts[value_counts <= rare_threshold].sum()  # レアカテゴリの行数 
            rare_ratio = (rare_rows / n) * 100
    
            if n > 0:
                mode_val = df[col].mode().iloc[0]
                mode_count = (df[col] == mode_val).sum()
                mode_pct = (mode_count / n) * 100
                mode_display = f"{mode_val} ({mode_count}件, {mode_pct:.1f}%)"
            else:
                mode_display = None
    
            stats.append({
                "変数": col,
                "n": n,
                "カテゴリ数": n_unique,
                "最頻値": mode_display,
                "ユニーク率": f"{unique_ratio:.1f}%",
                f"レアカテゴリ率(n≦{rare_threshold})": f"{rare_ratio:.1f}%",
                "欠損数": f"{n_missing} ({missing_pct:.1f}%)"
            })
        return pd.DataFrame(stats)

    if df_category.shape[1] > 0:
        cat_summary_df = cat_summary_stats(df_category)
        cat_summary_df.reset_index(drop=True, inplace=True)  # 行番号を連番に
        
        # --- カテゴリサマリー表 (DataTable) ---
        cat_summary_table = dash_table.DataTable(
            data=cat_summary_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in cat_summary_df.columns],
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": "bold"},
            style_cell={"textAlign": "center", "fontFamily": "Yu Gothic, Meiryo, sans-serif"}
        )
        
        # --- 行ごとの Dropdown を作成 ---
        dropdown_rows = []
        for idx, row in cat_summary_df.iterrows():
            var_name = row["変数"]
            categories = df_category[var_name].dropna().unique()
            n_unique = len(categories)
            n = df_category[var_name].count()
            unique_ratio = (n_unique / n) * 100 if n > 0 else 100
            
            # ユニーク率が70%以上ならスキップ（エンコード対象外）
            if unique_ratio >= 70:
                dropdown_rows.append(
                    dcc.Dropdown(
                        id={"type": "dynamic-dropdown", "col": var_name},
                        options=[{"label": "しない", "value": "しない"}],
                        value="しない",
                        style={"display": "none"}  # ← 非表示にする
                    )
                )
                continue
            # 欠損を除いて "O"/"X" だけならスキップ（エンコード対象外）
            if set(categories) <= {"O", "X"}:
                dropdown_rows.append(
                    dcc.Dropdown(
                        id={"type": "dynamic-dropdown", "col": var_name},
                        options=[{"label": "しない", "value": "しない"}],
                        value="しない",
                        style={"display": "none"}
                    )
                )
                continue    
            
            options = [
                {"label": "しない", "value": "しない"},
                {"label": "説明変数", "value": "説明変数"},
                {"label": "目的変数", "value": "目的変数"}
            ]
        
            dropdown_rows.append(
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "4px"},
                    children=[
                        html.Div(row["変数"], style={"flex": "0 0 20%", "textAlign": "right", "paddingRight": "10px"}),
                        dcc.Dropdown(
                            id={"type": "dynamic-dropdown", "col": var_name},
                            options=options,
                            value="しない",
                            clearable=False,
                            style={"flex": "0 0 40%"}
                        )
                    ]
                )
            )
        
        # ドロップダウンのコンテナにタイトルを追加
        visible_dropdowns = [
            d for d in dropdown_rows
            if not (isinstance(d, dcc.Dropdown) and d.style.get("display") == "none")
        ]
        if visible_dropdowns:  # 1つ以上ドロップダウンがあるときだけ表示
            cat_summary_dropdown = html.Div(
                children=[
                    html.H5("エンコード設定", style={"marginTop": "12px", "marginBottom": "8px"}),
                    html.Div([
                        html.Label("レアカテゴリをまとめる:", style={"color": "#34495e", "marginLeft": "10px"}),
                        dbc.Switch(
                            id="rare-category-switch",
                            value=False,  # 初期状態では無効
                            style={"marginLeft": "10px"}
                        )
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginTop": "20px"}),
                    html.Div(dropdown_rows)
                ]
            )
        else:
            cat_summary_dropdown = html.Div(
                children=[
                    html.Div([
                        dbc.Switch(id="rare-category-switch", value=False)
                    ], style={"display": "none"}),
                ]
            )
    else:
        cat_summary_table = ""       # カテゴリデータがないときは表示しない    
        cat_summary_dropdown = html.Div(
            children=[
                html.Div([
                    dbc.Switch(id="rare-category-switch", value=False)
                ], style={"display": "none"}),
            ]
        )

    return num_summary_table, cat_summary_table, cat_summary_dropdown

# エンコード設定
@app.callback(
    [Output("store-num-data", "data"),
     Output("store-class-data", "data"),
     Output("store-label-data", "data"),
     Output("impute-option", "style"),
     Output("store-original-categories", "data"),
     Output("store-rare-categories", "data")],
    [Input("store-original-data", "data"),
     Input("rare-category-switch", "value"),
     Input({"type": "dynamic-dropdown", "col": ALL}, "value"),
     Input("impute-mode", "value")],
    State({"type": "dynamic-dropdown", "col": ALL}, "id")
)
def encode_data(stored_data, rare_category_switch, dropdown_values, impute_method, dropdown_ids):
    if stored_data is None:
        return None, None, None, {"display": "none"}, None, None

    df = pd.DataFrame(stored_data["data"])
    df.columns = stored_data["columns"]

    # カテゴリ変数を抽出
    df_category = df.select_dtypes(exclude=[np.number])
    cat_summary_df = pd.DataFrame({"変数": df_category.columns})
    label_data = df_category.copy()
    if "行番号" in df.columns:
        num_data = df.select_dtypes(include=[np.number]).drop(columns=["行番号"], errors="ignore").copy()  # num_data から除外
        label_data.insert(0, "行番号", df["行番号"])  # label_data の先頭に追加
    else:
        num_data = df.select_dtypes(include=[np.number]).copy()
    class_columns = df_category.columns[df_category.apply(lambda col: col.dropna().isin(["O","X"]).all(), axis=0)]
    class_data = df_category[class_columns].copy()

    # レアカテゴリ統合のために df をコピーして加工
    df_processed = df.copy()

    # カテゴリ変数の欠損補完を適用
    for col in df_category.columns:
        if impute_method == "median_mode":
            if not df_processed[col].dropna().empty:
                mode_val = df_processed[col].mode().iloc[0]
                df_processed[col] = df_processed[col].fillna(mode_val)
        elif impute_method == "iterative_missing":
            df_processed[col] = df_processed[col].fillna("欠損")

    original_categories = {col: sorted(df_processed[col].dropna().unique().tolist()) for col in df_category.columns}

    option_map = { id_obj["col"]: value for id_obj, value in zip(dropdown_ids, dropdown_values) }

    n_total = len(df)
    rare_threshold = int(np.ceil(np.sqrt(n_total) / 5))
    rare_category_map = {}
     
    for col in cat_summary_df["変数"]:
        if col == "__dummy__": 
            continue  # ダミー列はスキップ
            
        option = option_map.get(col, "しない")
        rare_category_map[col] = []
     
        # レアカテゴリ統合
        if option in ["目的変数", "説明変数"] and rare_category_switch:
            value_counts = df[col].value_counts(dropna=True)
            rare_cats = value_counts[value_counts <= rare_threshold].index.tolist()
         
            # レアカテゴリが1つ以下 or 全カテゴリがレア → 統合スキップ
            if len(rare_cats) <= 1 or len(rare_cats) == len(value_counts):
                pass # 何もしない
            else:
                df_processed[col] = df[col].mask(df[col].isin(rare_cats), "その他")
                rare_category_map[col] = rare_cats

        # One-hot エンコード
        if option == "しない":
            continue  # 元の label_data に残すだけ
        elif option == "目的変数":
            # dropなし One-hot → 0/1 を X/O に変換
            dummies = pd.get_dummies(df_processed[col], prefix="", prefix_sep="", drop_first=False)
            dummies = dummies.astype(float)
            dummies.columns = [f"{col}[{c}]" for c in dummies.columns]
            dummies.loc[df_processed[col].isna(), :] = np.nan
            dummies = dummies.replace({0: "X", 1: "O"})
            class_data = pd.concat([class_data, dummies], axis=1)
        elif option == "説明変数":
            # drop なし one-hot
            dummies = pd.get_dummies(df_processed[col], prefix="", prefix_sep="", drop_first=False)
            dummies = dummies.astype(float)
            dummies.columns = [f"{col}[{c}]" for c in dummies.columns]
            dummies.loc[df_processed[col].isna(), :] = np.nan
            num_data = pd.concat([num_data, dummies], axis=1)
        else:
            # 不明な選択肢はスキップ
            continue

    # 数値データに欠損があるかどうかを判定
    if num_data.isnull().any().any():
        impute_style = {"display": "flex", "alignItems": "center", "marginTop": "10px"}
    else:
        impute_style = {"display": "none"}

    return (
        {"data": num_data.to_dict("records"), "columns": list(num_data.columns)},
        {"data": class_data.to_dict("records"), "columns": list(class_data.columns)},
        {"data": label_data.to_dict("records"), "columns": list(label_data.columns)},
        impute_style,
        {"data": original_categories},
        {"data": rare_category_map},
    )

@app.callback(
    [Output("corr-matrix", "figure"),
     Output("eta-cramers-matrix", "figure"),
     Output("vif-matrix", "figure"),
     Output("corr-top", "style"),
     Output("corr-bottom", "style"),
     Output("vif-accordion", "style")],
    [Input("store-original-data", "data")]
)
def upload_analysis(stored_data):
    if stored_data is None:
        return (
            default_fig("相関係数行列", ""),
            default_fig("相関比 / クラメールのV", ""),
            default_fig("多重共線性ヒートマップ", ""),
            {"display": "none"},
            {"display": "none"},
            {"display": "none"}
        )
    
    # DataFrameに復元
    df = pd.DataFrame(stored_data["data"])
    df.columns = stored_data["columns"]
    
    num_data = df.select_dtypes(include=[np.number]).drop(columns=["行番号"], errors="ignore").copy()  # 数値データのみを選択

    # 数値以外を抽出し、ユニーク率70%未満の列だけ残す
    cat_candidates = df.select_dtypes(exclude=[np.number])
    cat_data = pd.DataFrame()
    for col in cat_candidates.columns:
        n_data = df[col].count()
        if n_data > 0:
            unique_ratio = df[col].nunique() / n_data
            if unique_ratio < 0.7:
                cat_data[col] = df[col]

    if num_data.shape[1] > 0:
        # 相関行列の作成
        correlation_matrix = num_data.corr()
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale=[[0, "blue"], [0.5, "white"], [1, "red"]],
            zmin=-1,
            zmax=1,
            title="相関係数行列"
        )
        fig_corr.update_layout(
            #dragmode="pan",
            coloraxis=dict(colorbar_title="相関係数"),
            plot_bgcolor="darkgray",
            xaxis=dict(constrain="domain", showgrid=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(constrain="domain", showgrid=False)
        )
        fig_corr.update_traces(hovertemplate="<b>x: %{x}</b><br><b>y: %{y}</b><br><b>相関係数: %{z}</b><extra></extra>")
        corr_top = {"display": "block"}

        # VIF行列を計算する関数
        def calculate_vif(data):
            data_filled = data.fillna(data.mean())  # 平均値で補完
            features = data_filled.columns
            Xc = data_filled.values
            n = Xc.shape[1]
            model = Ridge(alpha=0.1)
            
            # データの標準化
            scaler = RobustScaler()
            Xc_standardized = scaler.fit_transform(Xc)
        
            rc = np.empty((n, n))
            vif_values = []
            r2_values = []
            X = np.copy(Xc_standardized)
            for i in range(n):
                y = np.copy(X[:, i])  # ターゲット変数
                X[:, i] = 1
        
                model.fit(X, y)
                y_pred = model.predict(X)
        
                score = r2_score(y, y_pred)    # R²スコアを計算
                if score < 1:
                    vif_values.append(1 / (1 - score))  # VIF値を保存
                else:
                    vif_values.append(np.nan)
                r2_values.append(score)  # R²値を保存
                rc[i] = model.coef_ #/ (np.linalg.norm(model.coef_) + 1e-15) * score
                X[:, i] = y  # 元に戻す
        
            # 回帰係数行列をpd.DataFrame形式に変換
            vif_matrix = pd.DataFrame(rc, index=features, columns=features).iloc[::-1, :]
            vif_table = pd.DataFrame({"説明変数": features, "VIF値": vif_values, "R²値": r2_values})
            vif_table = vif_table.sort_values(by="VIF値", ascending=False)
            return vif_matrix, vif_table
        
        # VIFを計算
        vif_matrix, vif_table = calculate_vif(num_data)
        
        # プロットを作成
        fig_vif = sp.make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "xy"}, {"type": "table"}]],
            subplot_titles=("多重共線性ヒートマップ", "VIF値(降順)")
        )
        
        # VIFヒートマップを追加
        heatmap = go.Heatmap(
            z=vif_matrix.values,
            x=vif_matrix.columns,
            y=vif_matrix.index,
            colorscale=[[0, "blue"], [0.5, "white"], [1, "red"]],
            zmin=-1,
            zmax=1,
            showscale=False,
            text=np.vectorize(lambda v: f"{v:.2f}")(vif_matrix.values),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        )
        
        # VIF値の表を追加
        table = go.Table(
            columnwidth=[2, 1, 1],
            header=dict(values=["説明変数", "VIF値", "R²値"], fill_color="lightgray", align="center"),
            cells=dict(values=[
                vif_table["説明変数"], 
                vif_table["VIF値"].round(2),
                vif_table["R²値"].round(3)
            ],align=["left", "right", "right"])
        )
    
        fig_vif.add_trace(heatmap, row=1, col=1)
        fig_vif.add_trace(table, row=1, col=2)
        #fig_vif.update_layout(dragmode="pan")
        vif_acc = {"display": "block"}
    else:
        fig_corr = default_fig("相関係数行列", "")
        fig_vif = default_fig("多重共線性ヒートマップ", "")
        corr_top = {"display": "none"}
        vif_acc = {"display": "none"}

    if cat_data.shape[1] > 0:
        # 相関比（カテゴリ vs 数値）
        def correlation_ratio(categories, values):
            # カテゴリと数値を横に結合して、両方欠損のない行だけ残す
            aligned = pd.concat([pd.Series(categories), pd.Series(values)], axis=1).dropna()
            categories = aligned.iloc[:, 0]
            values = aligned.iloc[:, 1]
    
            if len(values) == 0:
                return np.nan
    
            fcat, _ = pd.factorize(categories)
            cat_num = np.max(fcat) + 1
            y_avg = np.mean(values)
    
            numerator = 0
            denominator = np.sum((values - y_avg) ** 2)
    
            for i in range(cat_num):
                cat_values = values.iloc[np.argwhere(fcat == i).flatten()]
                if len(cat_values) > 0:
                    numerator += len(cat_values) * (np.mean(cat_values) - y_avg) ** 2
    
            return numerator / denominator if denominator != 0 else 0
    
        # クラメールのVを計算する関数
        def cramers_v(x, y):
            # 欠損は除外（カテゴリ化しない）
            aligned = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
            x = aligned.iloc[:, 0]
            y = aligned.iloc[:, 1]
    
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            r, k = confusion_matrix.shape
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))  #補正あり
            #return np.sqrt(phi2 / min(k-1, r-1))  #補正なし
    
        # 軸の定義（x:カテゴリ, y:数値＋カテゴリ）
        all_x = list(cat_data.columns)                           # カテゴリ
        all_y = list(num_data.columns) + list(cat_data.columns)  # 数値＋カテゴリ
        
        matrix = pd.DataFrame(index=all_y, columns=all_x)
    
        for x in all_x:  # 列:カテゴリ変数
            for y in all_y:  # 行:数値 or カテゴリ
                if y in num_data.columns:  # 数値 vs カテゴリ → 相関比
                    matrix.loc[y, x] = correlation_ratio(cat_data[x], num_data[y])
                else:  # カテゴリ vs カテゴリ → クラメールのV
                    matrix.loc[y, x] = cramers_v(cat_data[x], cat_data[y])
    
        matrix = matrix.astype(float)
    
        fig_cat = px.imshow(
            matrix,
            text_auto=".2f",
            color_continuous_scale=[[0, "white"], [1, "red"]],
            zmin=0,
            zmax=1,
            title="相関比 / クラメールのV"
        )
        fig_cat.update_layout(
            coloraxis=dict(colorbar_title="関連度"),
            plot_bgcolor="darkgray",
            xaxis=dict(side="bottom")
        )
        fig_cat.update_traces(
            hovertemplate="<b>x: %{x}</b><br><b>y: %{y}</b><br><b>関連度: %{z}</b><extra></extra>"
        )
        corr_bottom = {"display": "block"}
    else:
        fig_cat = default_fig("相関比 / クラメールのV", "")
        corr_bottom = {"display": "none"}

    return fig_corr, fig_cat, fig_vif, corr_top, corr_bottom, vif_acc

# コールバック: 相関係数行列クリックで散布図表示
@app.callback(
    Output('pair-plot', 'figure'),
    [Input('corr-matrix', 'clickData'),
    Input("store-num-data", "data"),
    Input("store-label-data", "data"),
    Input("color-toggle", "value")],
    State("label-variables", "value")
)
def update_plot(clickData, stored_data, label_data, color_toggle, label_variable):
    if clickData is None:
        return default_fig("データ分布","相関係数行列のセルをクリックで表示")

    data = pd.DataFrame(stored_data["data"])
    label_data = pd.DataFrame(label_data["data"])
    label_data = label_data.fillna("データなし")
    specific_labels = ["X", "データなし", "O"]  # 分類のラベル順序を定義
    # ラベルの順序と数値割り当てを作成
    if set(label_data[label_variable].unique()).issubset(specific_labels):  # 分類の場合
        label_mapping = {label: idx for idx, label in enumerate(specific_labels)}
    else:
        unique_labels = label_data[label_variable].unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = label_data[label_variable].map(label_mapping)  # 数値ラベルに変換
    
    point = clickData['points'][0]
    x_data = data[point['x']]  # クリックされたセルの列
    y_data = data[point['y']]  # クリックされたセルの行
    
    # プロットの色分け用設定
    if color_toggle and label_variable:
        color = numeric_labels  # 数値化されたラベルで色分け
    else:
        color = "blue"  # スイッチが無効の場合はデフォルト色

    hover_label = label_variable if label_variable else "ラベル"
    hovertemplate = (
        f"<b>{hover_label}:</b> "  # ラベル名を含める
        "%{text}<br><b>x:</b> %{x}<br><b>y:</b> %{y}<extra></extra>"
    )

    # 散布図
    scatter = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        name="散布図",
        marker=dict(color=color, colorscale="Bluered"),
        text=label_data.loc[data.index, label_variable] if label_variable else data.index + "行目",  # ラベル変数がない場合は行番号を表示
        hovertemplate=hovertemplate
    )
    # x軸のヒストグラム
    hist_x = go.Histogram(
        x=x_data,
        name="X軸ヒストグラム",
        marker=dict(color="lightblue"),
        yaxis="y2"  # 別のy軸を使用して配置
    )
    # y軸のヒストグラム
    hist_y = go.Histogram(
        y=y_data,
        name="Y軸ヒストグラム",
        marker=dict(color="lightpink"),
        xaxis="x2"  # 別のx軸を使用して配置
    )
    # 図の作成
    layout = go.Layout(
        title=f"データ分布: {point['y']} vs {point['x']}",
        xaxis=dict(title=point['x'],domain=[0, 0.8]),
        yaxis=dict(title=point['y'],domain=[0, 0.8]),
        xaxis2=dict(domain=[0.82, 1]),  # x軸上にヒストグラムを配置
        yaxis2=dict(domain=[0.82, 1]),  # y軸にヒストグラムを配置
        showlegend=False,
    )
    fig_pairplot = go.Figure(data=[scatter, hist_x, hist_y], layout=layout)
    fig_pairplot.update_layout(
        #dragmode="pan"
    )

    return fig_pairplot

# コールバック: 相関比/クラメールのVのセルクリックで図表示
@app.callback(
    Output('crosstab', 'figure'),
    [Input('eta-cramers-matrix', 'clickData'),
    Input("store-num-data", "data"),
    Input("store-label-data", "data"),
    Input("color-toggle", "value")],
    State("label-variables", "value")
)
def update_crosstab(clickData, num_data, cat_data, color_toggle, label_variable):
    if clickData is None:
        return default_fig("関連度プロット", "相関比/クラメールのVのセルをクリックで表示")

    num_df = pd.DataFrame(num_data["data"])
    raw_cat_df = pd.DataFrame(cat_data["data"])
    # ユニーク率を計算して 70% 未満の列だけ残す
    cat_df = pd.DataFrame()
    for col in raw_cat_df.columns:
        unique_ratio = raw_cat_df[col].nunique() / len(raw_cat_df)
        if unique_ratio < 0.7:   # ユニーク率が 70% 未満ならカテゴリとして採用
            cat_df[col] = raw_cat_df[col]

    point = clickData['points'][0]
    x_name = point['x']
    y_name = point['y']

    # ケース1: xがカテゴリ, yが数値 → 箱ひげ図
    if x_name in cat_df.columns and y_name in num_df.columns:
        x_series = cat_df[x_name].astype(object).fillna("欠損")  # 欠損をカテゴリ化
        y_series = num_df[y_name]                                # 数値

        # 数値側の欠損は描画できないので除外
        mask = ~y_series.isna()
        x_plot = x_series[mask]
        y_plot = y_series[mask]

        # カテゴリ順で欠損を最後に並べる
        uniques = pd.Index(x_plot.unique())
        ordered = [c for c in uniques if c != "欠損"] + (["欠損"] if "欠損" in uniques else [])

        fig = go.Figure()
        fig.add_trace(go.Box(
            x=x_plot,
            y=y_plot,
            boxpoints="all",
            jitter=0.3,
            marker_color="blue",
        ))
        fig.update_layout(
            title=f"箱ひげ図: {y_name} vs {x_name}",
            xaxis_title=x_name,
            yaxis_title=y_name,
            xaxis=dict(categoryorder="array", categoryarray=ordered)
        )
        return fig

    # ケース2: xもyもカテゴリ → クロス集計ヒートマップ
    elif x_name in cat_df.columns and y_name in cat_df.columns:
        x_series = cat_df[x_name].fillna("欠損")
        y_series = cat_df[y_name].fillna("欠損")

        crosstab = pd.crosstab(y_series, x_series)

        # 欠損を最後に並べ替え
        if "欠損" in crosstab.columns:
            cols = [c for c in crosstab.columns if c != "欠損"] + ["欠損"]
            crosstab = crosstab[cols]
        if "欠損" in crosstab.index:
            rows = [r for r in crosstab.index if r != "欠損"] + ["欠損"]
            crosstab = crosstab.loc[rows]

        fig = px.imshow(
            crosstab,
            text_auto=True,
            color_continuous_scale=[[0, "white"], [1, "red"]],
            aspect="auto",
            zmin=0, zmax=len(cat_df),
            title=f"クロス集計: {y_name} vs {x_name}"
        )
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name,
            coloraxis=dict(colorbar_title="件数")
        )
        return fig

    # その他（想定外）
    else:
        return default_fig("関連度プロット", "この組み合わせには対応していません")

# エンコード変数のグループ化用関数
def group_encoded_columns(columns):
    groups = {}
    for col in columns:
        if "[" in col and col.endswith("]"):
            base = col.split("[")[0]
            groups.setdefault(base, []).append(col)
        else:
            groups.setdefault(col, [col])
    return groups

# コールバック: チェックボックスオプションを生成し、選択を制御
@app.callback(
    [Output("option-explanatory", "options"),
     Output("option-target", "options"),
     Output("option-class", "options"),
     Output("label-variables", "options")],
    [Input("store-num-data", "data"),
     Input("store-class-data", "data"),
     Input("store-label-data", "data"),
     Input("option-explanatory", "value"),
     Input("option-target", "value"),
     Input("impute-mode", "value")]
)
def update_display_options(stored_data, stored_class_data, stored_label_data,
                           explanatory_values, target_values, impute_method):
    if stored_data is None:
        dummy = [{"label": "データなし", "value": "none"}]
        return dummy, dummy, dummy, dummy

    # 数値データ（エンコード後）
    num_data = pd.DataFrame(stored_data["data"])
    groups = group_encoded_columns(num_data.columns)

    # 欠損判定（元変数単位で判定）
    columns = []
    for base, cols in groups.items():
        has_na = num_data[cols].isnull().any().any()
        # impute-mode が none のときだけ欠損列を disabled にする
        disabled_flag = (impute_method == "none") and has_na
        columns.append({"label": base, "value": base, "disabled": disabled_flag})

    # 相互排他設定
    explanatory_options = [
        {"label": col["label"], "value": col["value"],
         "disabled": col["disabled"] or col["value"] in target_values}
        for col in columns
    ]
    target_options = [  # 目的変数（列名に "[" を含まない）
        {"label": col, "value": col, "disabled": col in explanatory_values}
        for col in num_data.columns if "[" not in col
    ]

    # クラスデータ
    class_data = pd.DataFrame(stored_class_data["data"])
    class_groups = group_encoded_columns(class_data.columns)
    option_class = [{"label": k, "value": k} for k in class_groups.keys()]

    # ラベルデータ
    label_data = pd.DataFrame(stored_label_data["data"])
    columns = [{"label": col, "value": col} for col in label_data.columns]
    option_label = [{"label": col["label"], "value": col["value"]} for col in columns]

    return explanatory_options, target_options, option_class, option_label

@app.callback(
    Output("label-variables", "value"),
    Input("store-label-data", "data")
)
def set_initial_label(stored_label_data):
    if stored_label_data is None:
        return None

    label_data = pd.DataFrame(stored_label_data["data"])
    initial_label_value = "行番号" if "行番号" in label_data.columns else None
    return initial_label_value

#選択肢→解析する変数に展開
@app.callback(
    [Output("explanatory-variables", "value"),
     Output("target-variables", "value"),
     Output("class-variables", "value")],
    [Input("option-explanatory", "value"),
     Input("option-target", "value"),
     Input("option-class", "value")],
    [State("store-num-data", "data"),
     State("store-class-data", "data"),
     State("store-label-data", "data")]
)
def expand_selected(display_explanatory, display_target, display_class,
                    stored_data, stored_class_data, stored_label_data):
    def expand(values, groups):
        expanded = []
        if values is None:
            return []
        for v in values:
            expanded += groups.get(v, [v])
        return expanded

    if not stored_data or "data" not in stored_data:
            return [], [], []
    if not stored_class_data or "data" not in stored_class_data:
        return [], [], []
    num_data = pd.DataFrame(stored_data["data"])
    groups_num = group_encoded_columns(num_data.columns)

    class_data = pd.DataFrame(stored_class_data["data"])
    groups_class = group_encoded_columns(class_data.columns)

    return (
        expand(display_explanatory, groups_num),
        expand(display_target, groups_num),
        expand(display_class, groups_class)
    )

# コールバック: 全選択ボタン
@app.callback(
    [Output("option-explanatory", "value"),
     Output("option-target", "value"),
     Output("option-class", "value")],
    [Input("select-all-explanatory", "n_clicks"),
     Input("select-all-target", "n_clicks"),
     Input("select-all-class", "n_clicks"),
     Input("store-num-data", "data"),
     Input("impute-mode", "value")],
    [State("option-explanatory", "options"),
     State("option-target", "options"),
     State("option-class", "options"),
     State("option-explanatory", "value"),
     State("option-target", "value"),
     State("option-class", "value")]
)
def handle_variables(n_clicks_explanatory, n_clicks_target, n_clicks_class, stored_data, impute_method,
                     explanatory_options, target_options, class_options, explanatory_values, target_values, class_values):
    # どのボタンが押されたかを判定
    triggered_id = ctx.triggered_id

    if stored_data is None:
        return [], [], []
    if triggered_id == "store-num-data":  # 選択肢が変わったら選択リセット
        return [], [], []
    
    # 欠損のある変数だけチェックを外す処理
    if impute_method == "none":
        num_data = pd.DataFrame(stored_data["data"])
        groups = group_encoded_columns(num_data.columns)

        # 欠損のある変数を除外
        explanatory_values = [
            v for v in explanatory_values
            if not num_data[groups[v]].isnull().any().any()
        ]
    def filter_valid_values(options, current_values):
        # オプションから選択可能な値のみ抽出し、無効な選択を削除
        valid_values = [opt["value"] for opt in options if not opt.get("disabled", False)]
        return [value for value in current_values if value in valid_values]

    # 現在の選択値をフィルタリングして有効な値のみ残す
    explanatory_values = filter_valid_values(explanatory_options, explanatory_values)
    target_values = filter_valid_values(target_options, target_values)

    # 全選択ボタンの処理
    if triggered_id == "select-all-explanatory":
        if len(explanatory_values) == len([opt for opt in explanatory_options if not opt.get("disabled", False)]):
            return [], target_values, class_values  # 全て選択されている場合は解除
        else:
            return [opt["value"] for opt in explanatory_options if not opt.get("disabled", False)], target_values, class_values
    elif triggered_id == "select-all-target":
        if len(target_values) == len([opt for opt in target_options if not opt.get("disabled", False)]):
            return explanatory_values, [], class_values  # 全て選択されている場合は解除
        else:
            return explanatory_values, [opt["value"] for opt in target_options if not opt.get("disabled", False)], class_values
    elif triggered_id == "select-all-class":
        if len(class_values) == len(class_options):
            return explanatory_values, target_values, []  # 全て選択されている場合は解除
        else:
            return explanatory_values, target_values, [opt["value"] for opt in class_options]

    return explanatory_values, target_values, class_values

# コールバック: GP解析
@app.callback(
    [Output({"type": "saved-cv-scores", "index": ALL}, "data"),
     Output({"type": "stored-model", "index": ALL}, "data"),
     Output({"type": "result-message", "index": ALL}, "children"),
     Output({"type": "scatter-plot", "index": ALL}, "figure"),
     Output({"type": "feature-importance-plot", "index": ALL}, "figure"),
     Output({"type": "cross-validation-plot", "index": ALL}, "figure"),
     Output({"type": "cv-export-table", "index": ALL}, "data")],
    [Input("dynamic-tabs", "children"),
     Input({"type": "threshold-slider", "index": ALL}, "value")],
    [State("color-toggle", "value"),
     State("store-num-data", "data"),
     State("store-class-data", "data"),
     State("store-label-data", "data"),
     State("explanatory-variables", "value"),
     State("target-variables", "value"),
     State("class-variables", "value"),
     State("label-variables", "value"),
     State({"type": "stored-model", "index": ALL}, "data"),
     State("impute-mode", "value"),
     State("tuning-type", "value")],
    prevent_initial_call=True
)
def update_models(tabs, threshold_list, color_toggle, stored_data, stored_class_data, label_data,  
                  explanatory_variables, target_variables, class_variables, label_variable, 
                  saved_model, impute_method, tuning_type):

    total_variables = target_variables + class_variables
    triggered_id = ctx.triggered[0]['prop_id']
    
    if stored_data is None or (not target_variables and not class_variables):
        # データがない場合、各目的変数にデフォルトを返す
        return [
            [{"cv_score": None}] * len(total_variables),
            [{"model": None}] * len(total_variables),
            [html.Div("目的変数が選択されていません。", style={"color": "red"})] * len(total_variables),
            [default_fig("実測値 vs 予測値", "")] * len(total_variables),
            [default_fig("特徴量の重要度", "")] * len(total_variables),
            [default_fig("CV 実測値 vs 予測値", "")] * len(total_variables),
            [[]] * len(total_variables),
        ]
    elif not explanatory_variables:
        return [
            [{"cv_score": None}] * len(total_variables),
            [{"model": None}] * len(total_variables),
            [html.Div("説明変数が選択されていません。", style={"color": "red"})] * len(total_variables),
            [default_fig("実測値 vs 予測値", "")] * len(total_variables),
            [default_fig("特徴量の重要度", "")] * len(total_variables),
            [default_fig("CV 実測値 vs 予測値", "")] * len(total_variables),
            [[]] * len(total_variables),
        ]      

    try:
        data = pd.DataFrame(stored_data["data"])
        class_data = pd.concat([data[explanatory_variables], pd.DataFrame(stored_class_data["data"])], axis=1)
        label_data = pd.DataFrame(label_data["data"]).fillna("データなし")

        # 結果を格納するリスト
        saved_cv_scores = []
        model_data = []
        result_messages = []
        scatter_figures = []
        importance_figures = []
        cv_figures = []
        cv_export_data = []

        # 各目的変数に対して処理(回帰)
        for target in target_variables:
            if "dynamic-tabs" in triggered_id:
                dropped_data = data.dropna(subset=[target])
                X = dropped_data[explanatory_variables].values
                n_features = X.shape[1]
                y = dropped_data[target].values
                n_fold = int(np.floor(min(len(y), 10, 3 + 100 / np.sqrt(len(y)))))
                n_restart = int(np.ceil(30 * 10 ** (-0.01 * len(y))))
    
                # --- 指標(Scorer)の選択 ---
                if tuning_type == "precise":
                    current_scorer = rmse_scorer
                else:
                    current_scorer = huber_scorer
    
                pipe = Pipeline([
                    ("prep", Preprocessor(impute_method=impute_method, scale=True)),
                    ("gpr", GaussianProcessRegressor(alpha=1e-3, n_restarts_optimizer=n_restart, random_state=0))
                ])
            
                # ガウス過程回帰モデルの作成
                if load_model(saved_model, target) is None:
                    param_grid = {
                        'gpr__kernel': [
                            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_features),nu=0.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_features), nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_features), nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                            #ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                            #ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(n_features)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                            #ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                            ConstantKernel(1.0, (1e-3, 1e3)) * (Matern(np.ones(n_features), nu=1.5) + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 3)),
                        ]
                    }
                    gpr_grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=n_fold, scoring=current_scorer, n_jobs=-1)
                    gpr_grid.fit(X, y)
                    gpr_opt = gpr_grid.best_estimator_
                    
                    print(gpr_opt.named_steps["gpr"].kernel_)
                else:
                    gpr_opt = load_model(saved_model, target)
                
                y_pred, y_err = gpr_opt.predict(X, return_std=True)  # 平均と標準偏差を取得
    
                # プロットの色分け用設定
                specific_labels = ["X", "データなし", "O"]  # 分類のラベル順序を定義
                # ラベルの順序と数値割り当てを作成
                if set(label_data[label_variable].unique()).issubset(specific_labels):  # 分類の場合
                    label_mapping = {label: idx for idx, label in enumerate(specific_labels)}
                else:
                    unique_labels = label_data[label_variable].unique()
                    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                numeric_labels = label_data[label_variable].map(label_mapping)  # 数値ラベルに変換
                numeric_labels = numeric_labels.loc[dropped_data.index]
        
                if color_toggle and label_variable:
                    color = numeric_labels  # 数値化されたラベルで色分け
                else:
                    color = "blue"  # スイッチが無効の場合はデフォルト色
    
                hover_label = label_variable if label_variable else "ラベル"
                hovertemplate = (
                    f"<b>{hover_label}:</b> "  # ラベル名を含める
                    "%{text}<br><b>実測値:</b> %{x}<br><b>予測値:</b> %{y}<extra></extra>"
                )
    
                # 散布図
                scatter_fig = go.Figure()
                scatter_fig.add_trace(
                    go.Scatter(
                        x=y,
                        y=y_pred,
                        mode="markers",
                        name="μ±1.96σ",
                        error_y=dict(
                            type="data",
                            array=y_err * 1.96,
                            visible=True,
                            thickness=0.4,
                            width=4,
                            color="rgba(0, 0, 0, 0.3)"
                        ),
                        marker=dict(color=color, colorscale="Bluered"),
                        text=label_data.loc[dropped_data.index, label_variable] if label_variable else dropped_data.index + "行目",  # ラベル変数がない場合は行番号を表示
                        hovertemplate=hovertemplate
                    )
                )
                
                scatter_fig = scatter_layout(scatter_fig, y, y_pred, f"{target} (GPR)")
                scatter_figures.append(scatter_fig)
    
                # --- PDP 計算 ---
                X_scaled = gpr_opt.named_steps["prep"].transform(X)
                feature_names = explanatory_variables
                
                # 基準となる平均値
                overall_mean_pred = float(gpr_opt.named_steps["gpr"].predict(X_scaled).mean())
                
                pdp_results = []
                pdp_importance = []
                
                for i, feat in enumerate(feature_names):
                    # 1. 元の生データからユニークな値を取得（逆算による誤差を防ぐ）
                    raw_grid_values = np.unique(dropped_data[feat].values)
                    
                    if len(raw_grid_values) < 2:
                        pdp_importance.append(0.0)
                        continue

                    # 2. PDP用のスケール済みグリッド値を作成
                    # 他の特徴量は1行目のデータを使い、対象列だけ生データを入れてtransform
                    dummy_df = pd.DataFrame(np.tile(X.iloc[0].values if isinstance(X, pd.DataFrame) else X[0], 
                                            (len(raw_grid_values), 1)), columns=feature_names)
                    dummy_df[feat] = raw_grid_values
                    grid_values_scaled = gpr_opt.named_steps["prep"].transform(dummy_df.values)[:, i]
                
                    # 3. PDP を計算
                    pdp_values = []
                    for v_raw in raw_grid_values:
                        X_temp_raw = dropped_data[explanatory_variables].copy()
                        X_temp_raw[feat] = v_raw

                        preds = gpr_opt.predict(X_temp_raw)
                        pdp_values.append(preds.mean())
                    pdp_values = np.asarray(pdp_values)
                
                    pdp_contrib = pdp_values - overall_mean_pred
                
                    # 4. 密度加重重要度
                    actual_data_scaled = X_scaled[:, i]
                    counts, _ = np.histogram(actual_data_scaled, bins=len(grid_values_scaled))
                    weights = counts / (counts.sum() + 1e-9)
                    weighted_importance = np.sum(np.abs(pdp_contrib) * weights)
                    pdp_importance.append(weighted_importance)
                
                    # 5. 色の正規化（表示範囲で 0~1 にきっちり収める）
                    v_min, v_max = grid_values_scaled.min(), grid_values_scaled.max()
                    denom = v_max - v_min + 1e-9
                
                    for g_raw, p_contrib, v_s in zip(raw_grid_values, pdp_contrib, grid_values_scaled):
                        s_norm = (v_s - v_min) / denom
                        pdp_results.append({
                            "feature": feat,
                            "feature_value_original": g_raw,
                            "feature_value_scaled": s_norm, # 青〜赤のレンジを使い切る
                            "pdp_y": p_contrib
                        })
                
                df_pdp = pd.DataFrame(pdp_results)
                
                # --- 特徴量ごとの y 軸位置 ---
                feature_order = {f: i for i, f in enumerate(feature_names)}
                df_pdp["y_jittered"] = df_pdp.apply(
                    lambda row: feature_order[row["feature"]] + (row["feature_value_scaled"] - 0.5) * 0.5,
                    axis=1
                )
                
                # --- 図の作成 ---
                importance_fig = sp.make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.4, 0.6],
                    subplot_titles=["寄与度 (平均絶対偏差)", "寄与度プロファイル"],
                    shared_yaxes=True,
                    horizontal_spacing=0.05
                )
                
                # 左：棒グラフ
                importance_fig.add_trace(go.Bar(
                    x=pdp_importance,
                    y=list(feature_order.values()),
                    orientation="h",
                    #marker=dict(color="royalblue"),
                    name="寄与度",
                ), row=1, col=1)
                
                # 右：散布図
                importance_fig.add_trace(go.Scatter(
                    x=df_pdp["pdp_y"],
                    y=df_pdp["y_jittered"],
                    mode="markers",
                    marker=dict(
                        color=df_pdp["feature_value_scaled"],
                        colorscale="Bluered",
                        showscale=True,
                        size=6,
                        opacity=0.7,
                        colorbar=dict(
                            title="特徴量の値",
                            tickvals=[0, 1],
                            ticktext=["小", "大"],
                            len=0.8,
                            thickness=15
                        )
                    ),
                    customdata=np.stack([df_pdp["feature"], df_pdp["feature_value_original"]], axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "寄与度(偏差): %{x:.3f}<br>"
                                  "特徴量の値: %{customdata[1]:.3f}<extra></extra>",
                    name="PDP分布"
                ), row=1, col=2)
                
                importance_fig.update_layout(
                    title="特徴量の重要度 (GPR)",
                    showlegend=False,
                )
    
                importance_fig.update_xaxes(title_text="寄与度", row=1, col=1)
                importance_fig.update_xaxes(title_text="寄与度", row=1, col=2)
                
                importance_fig.update_yaxes(
                    title_text="特徴量",
                    tickvals=list(feature_order.values()),
                    ticktext=list(feature_order.keys()),
                    row=1, col=1
                )
                importance_fig.update_yaxes(
                    title_text="特徴量",
                    tickvals=list(feature_order.values()),
                    ticktext=list(feature_order.keys()),
                    row=1, col=2
                )
                
                importance_figures.append(importance_fig)
    
                # クロスバリデーションプロット
                kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)
                all_y_test, all_y_pred = [], []
                cv_fig = go.Figure()
                fold_colors = generate_fold_colors(n_fold)
                df_oof = pd.DataFrame(index=data.index, columns=["fold", "y_true", "y_pred"])
                best_kernel = gpr_opt.named_steps["gpr"].kernel_
                
                pipe_cv = Pipeline([
                    ("prep", Preprocessor(impute_method=impute_method, scale=True)),
                    ("gpr", GaussianProcessRegressor(kernel=best_kernel, alpha=1e-6, random_state=0))
                ])
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
    
                    pipe_cv.fit(X_train, y_train)
                    y_pred_fold, y_err_fold = pipe_cv.predict(X_test, return_std=True)
                    
                    all_y_test.extend(y_test)
                    all_y_pred.extend(y_pred_fold)
                    hovertemplate = (
                        f"<b>Fold {fold + 1}</b> "
                        "<br><b>実測値:</b> %{x}<br><b>予測値:</b> %{y}<extra></extra>"
                    )
    
                    cv_fig.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred_fold,
                        mode="markers",
                        name=f"Fold {fold + 1}",
                        error_y=dict(
                            type="data",
                            array=y_err_fold * 1.96,
                            visible=True,
                            thickness=0.4,
                            width=4,
                            color="rgba(0, 0, 0, 0.3)"
                        ),
                        marker=dict(color=fold_colors[fold]),
                        hovertemplate=hovertemplate
                    ))
                    
                    df_oof.loc[dropped_data.index[test_idx], "fold"] = fold + 1
                    df_oof.loc[dropped_data.index[test_idx], "y_true"] = y_test.astype(float)
                    df_oof.loc[dropped_data.index[test_idx], "y_pred"] = y_pred_fold.astype(float)
    
                cv_fig = scatter_layout(cv_fig, all_y_test, all_y_pred, f"{n_fold}-fold CV {target} (GPR)")
                cv_figures.append(cv_fig)
                
                df_oof["行数"] = df_oof.index + 1
                df_oof = df_oof[["行数", "fold", "y_true", "y_pred"]]
                cv_export_data.append(df_oof.to_dict("records"))
    
                # 結果メッセージ
                r2 = concordance_corrcoef(all_y_test, all_y_pred)  #r2_score(all_y_test, all_y_pred)
                mae = mean_absolute_error(all_y_test, all_y_pred)
                saved_cv_scores.append({"target": target, "cv_score": r2})
                result_message = html.Div([
                    html.P(f"検証データへのモデル一致度: {r2:.5f},　平均的な誤差: {mae:.3f}", style={"color": "#3498db", "fontWeight": "bold"})
                ])
                result_messages.append(result_message)
                model_data.append({"name": target, "model": base64.b64encode(pickle.dumps(gpr_opt)).decode('utf-8')})
            else:
                scatter_figures.append(no_update)
                importance_figures.append(no_update)
                cv_figures.append(no_update)
                cv_export_data.append(no_update)
                saved_cv_scores.append(no_update)
                result_messages.append(no_update)
                model_data.append(no_update)
                
                
        # 各目的変数に対して処理(分類)
        for c_target in class_variables:
            dropped_data = class_data.dropna(subset=[c_target]).copy()
            dropped_data[c_target] = dropped_data[c_target].map({"O": 1, "X": 0}).astype(int)
            X = dropped_data[explanatory_variables].values
            y = dropped_data[c_target].to_numpy()
            min_class_count = np.min(np.bincount(y))
            n_fold = int(np.floor(min(min_class_count, 10, 3 + 100 / np.sqrt(len(y)))))
            gamma_base = 1.0 / X.shape[1]
            #cv = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=1, random_state=0)
            n_repeats = int(np.ceil(5 / (1 + (len(y) / 400))))

            if "dynamic-tabs" in triggered_id or load_model(saved_model, c_target) is None:
                '''
                # BayesSearchCVでSVMハイパーパラメータ探索                
                pipe = Pipeline([
                    ("prep", Preprocessor(impute_method=impute_method, scale=True)),
                    ("svm", SVC(probability=True, random_state=0))
                ])
                param_grid = {
                    "svm__C": (0.1, 50.0, "log-uniform"),
                    "svm__gamma": (gamma_base * 0.1, gamma_base * 10.0, "log-uniform"),
                    "svm__kernel": ["rbf"],
                    "svm__class_weight": ["balanced"]#, None]
                }
            
                bayes = BayesSearchCV(
                    estimator=pipe,
                    search_spaces=param_grid,
                    n_iter=100,
                    scoring="accuracy",
                    cv=cv,
                    random_state=0,
                    refit=True
                )
                bayes.fit(X, y)
                svm_opt = bayes.best_estimator_
                '''
                # --- Optuna Objective ---
                def objective(trial):
                    # パラメータサンプリング
                    c = trial.suggest_float("svm__C", 0.1, 50.0, log=True)
                    gamma = trial.suggest_float("svm__gamma", gamma_base * 0.1, gamma_base * 10.0, log=True)
                    cw = trial.suggest_categorical("svm__class_weight", [None, "balanced"])
                    
                    # パイプライン構成
                    pipe = Pipeline([
                        ("prep", Preprocessor(impute_method=impute_method, scale=True)),
                        ("svm", SVC(C=c, gamma=gamma, class_weight=cw, probability=True, random_state=0))
                    ])
    
                    # CVの実行 (RepeatedStratifiedKFoldで安定性を確保)
                    repeat_scores = []
                    for r in range(n_repeats):
                        # シードを変えて繰り返す
                        cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=r)
                        oof_probs = np.zeros(len(y))
                        
                        for train_idx, test_idx in cv.split(X, y):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            model = Pipeline(pipe.steps)
                            model.fit(X_train, y_train)
                            # 確率予測を取得
                            oof_probs[test_idx] = model.predict_proba(X_test)[:, 1]
    
                        # --- スコア付け ---
                        if tuning_type == "precise":
                            score = logloss(y, oof_probs)
                        else:
                            score = brier_score(y, oof_probs)
                            
                        repeat_scores.append(score)
    
                    # ---- mu と sigma の計算 ----
                    mu = np.mean(repeat_scores)
                    sigma = np.std(repeat_scores)
                    # λ（安定性の重み）: データが少ないほどバラつきを嫌う
                    lambda_ = 1 / (1 + len(y) / 100)
                    
                    return mu + lambda_ * sigma
                
                study = optuna.create_study(direction="minimize", sampler=sampler)
                n_jobs = int(np.ceil(os.cpu_count() * 0.6))
                study.optimize(objective, n_trials=100, n_jobs=n_jobs)
                
                # 最良パラメータで全データ再学習
                best_params = {k.replace("svm__", ""): v for k, v in study.best_params.items()}
                final_pipe = Pipeline([
                    ("prep", Preprocessor(impute_method=impute_method, scale=True)),
                    ("svm", SVC(**best_params, probability=True, random_state=0))
                ])
                final_pipe.fit(X, y)
                svm_opt = final_pipe
            else:
                svm_opt = load_model(saved_model, c_target)
                
            y_pred = svm_opt.predict_proba(X)[:, 1]

            threshold = threshold_list[total_variables.index(c_target)]
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            svm = svm_opt.named_steps["svm"]
            X_scaled = svm_opt.named_steps["prep"].transform(X)
            X_scaled = np.asarray(X_scaled)

            if "dynamic-tabs" in triggered_id:
                # 特徴量の重要度
                # 1. 準備：前処理済みのデータと、SVMモデル単体を取り出す
                prep = svm_opt.named_steps["prep"]
                svm_model = svm_opt.named_steps["svm"]
                X_transformed = np.asarray(prep.transform(X))
                feature_names = explanatory_variables 
                
                # 2. 基準値（確率の平均）
                pos_idx = list(svm_model.classes_).index(1)
                overall_mean_prob = float(svm_model.predict_proba(X_transformed)[:, pos_idx].mean())
                
                pdp_results = []
                pdp_importance = []
                
                for i, feat in enumerate(feature_names):
                    # 1. 元の生データからユニークな値を取得
                    raw_grid_values = np.unique(dropped_data[feat].values)
                    
                    if len(raw_grid_values) < 2:
                        pdp_importance.append(0.0)
                        continue

                    # 2. PDP用の「スケール済み」グリッド値を作成
                    # prep.transformができるように一時的なDataFrameを作る
                    dummy_df = pd.DataFrame(np.tile(X_transformed[0], (len(raw_grid_values), 1)), columns=feature_names)
                    for j, f in enumerate(feature_names):
                        # 他の特徴量は適当な値（変換済みの1行目）で埋め、対象列だけ生データを入れる
                        if f == feat:
                            dummy_df[f] = raw_grid_values
                        else:
                            pass
                    
                    # prep.transformを通した後の該当カラムの値をgrid_valuesとして使う
                    grid_values_scaled = prep.transform(dummy_df.values)[:, i]
                
                    # 3. 自前でPDPを計算
                    pdp_values = []
                    for v_scaled in grid_values_scaled:
                        X_tmp = X_transformed.copy()
                        X_tmp[:, i] = v_scaled
                        probs = svm_model.predict_proba(X_tmp)[:, pos_idx]
                        pdp_values.append(probs.mean())
                    pdp_values = np.asarray(pdp_values)
                
                    pdp_contrib = pdp_values - overall_mean_prob
                
                    # 4. 密度加重重要度の計算
                    actual_data_scaled = X_transformed[:, i]
                    counts, _ = np.histogram(actual_data_scaled, bins=len(grid_values_scaled))
                    weights = counts / (counts.sum() + 1e-9)
                    weighted_importance = np.sum(np.abs(pdp_contrib) * weights)
                    pdp_importance.append(weighted_importance)
                
                    # 5. 散布図プロット用の正規化（色の指定用：0~1）
                    v_min, v_max = grid_values_scaled.min(), grid_values_scaled.max()
                    norm_scaled_values = (grid_values_scaled - v_min) / (v_max - v_min + 1e-9)
                
                    for g_raw, p_contrib, s_norm in zip(raw_grid_values, pdp_contrib, norm_scaled_values):
                        pdp_results.append({
                            "feature": feat,
                            "feature_value_original": g_raw,  # ここに生の値を直接入れる
                            "feature_value_scaled": s_norm,   # ここは色のための0~1
                            "pdp_y": p_contrib
                        })
                
                df_pdp = pd.DataFrame(pdp_results)
                # --- 特徴量ごとのy軸位置 ---
                feature_order = {f: i for i, f in enumerate(feature_names)}
                df_pdp["y_jittered"] = df_pdp.apply(
                    lambda row: feature_order[row["feature"]] + (row["feature_value_scaled"] - 0.5) * 0.5, 
                    axis=1
                )
                
                # --- 図の作成 ---
                importance_fig = sp.make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.4, 0.6],
                    subplot_titles=["寄与度 (平均絶対偏差)", "寄与度プロファイル"],
                    shared_yaxes=True,
                    horizontal_spacing=0.05
                )
                
                # 左：棒グラフ
                importance_fig.add_trace(go.Bar(
                    x=pdp_importance,
                    y=list(feature_order.values()),
                    orientation="h",
                    #marker=dict(color="royalblue"),
                    name="寄与度",
                ), row=1, col=1)
                
                # 右：散布図
                importance_fig.add_trace(go.Scatter(
                    x=df_pdp["pdp_y"],
                    y=df_pdp["y_jittered"],
                    mode="markers",
                    marker=dict(
                        color=df_pdp["feature_value_scaled"],
                        colorscale="Bluered",
                        showscale=True,
                        size=6,
                        opacity=0.7,
                        colorbar=dict(
                            title="特徴量の値",
                            tickvals=[0, 1],
                            ticktext=["小", "大"],
                            len=0.8,
                            thickness=15
                        )
                    ),
                    customdata=np.stack([df_pdp["feature"], df_pdp["feature_value_original"]], axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "寄与度(偏差): %{x:.3f}<br>"
                                  "特徴量の値: %{customdata[1]:.3f}<extra></extra>",
                    name="PDP分布"
                ), row=1, col=2)
                
                importance_fig.update_layout(
                    title="特徴量の重要度 (SVM)",
                    showlegend=False,
                )
    
                importance_fig.update_xaxes(title_text="寄与度", row=1, col=1)
                importance_fig.update_xaxes(title_text="寄与度", row=1, col=2)
                
                importance_fig.update_yaxes(
                    title_text="特徴量",
                    tickvals=list(feature_order.values()),
                    ticktext=list(feature_order.keys()),
                    row=1, col=1
                )
                importance_fig.update_yaxes(
                    title_text="特徴量",
                    tickvals=list(feature_order.values()),
                    ticktext=list(feature_order.keys()),
                    row=1, col=2
                )
                
                importance_figures.append(importance_fig)
            else:
                importance_figures.append(no_update)

            # プロットの色分け用設定
            specific_labels = ["X", "データなし", "O"]  # 分類のラベル順序を定義
            # ラベルの順序と数値割り当てを作成
            if set(label_data[label_variable].unique()).issubset(specific_labels):  # 分類の場合
                label_mapping = {label: idx for idx, label in enumerate(specific_labels)}
            else:
                unique_labels = label_data[label_variable].unique()
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = label_data[label_variable].map(label_mapping)  # 数値ラベルに変換
            numeric_labels = numeric_labels.loc[dropped_data.index]
    
            if color_toggle and label_variable:
                color = numeric_labels  # 数値化されたラベルで色分け
            else:
                color = "blue"  # スイッチが無効の場合はデフォルト色

            hover_label = label_variable if label_variable else "ラベル"
            y_labels = ["O" if val == 1 else "X" for val in y]
            hovertemplate = (
                f"<b>{hover_label}:</b> "  # ラベル名を含める
                "%{text}<br><b>実測値:</b> %{customdata}<br><b>予測陽性確率:</b> %{y}<extra></extra>"
            )

            # 散布図: 実測値 (x軸) vs 予測確率 (y軸)
            sigmoid_fig = go.Figure()
            
            jitter_width = 0.2     # 実測値 y に少しランダムノイズを追加
            np.random.seed(0)
            y_jitter = y + np.random.uniform(-jitter_width, jitter_width, size=len(y))
            
            sigmoid_fig.add_trace(
                go.Scatter(
                    x=y_jitter,  # 実測値
                    y=y_pred,  # 予測確率
                    customdata=y_labels,
                    mode="markers",
                    name="全サンプル",
                    marker=dict(color=color, colorscale="Bluered"),
                    text=label_data.loc[dropped_data.index, label_variable] if label_variable else dropped_data.index + "行目",
                    hovertemplate=hovertemplate,
                )
            )
            
            # 閾値の線
            sigmoid_fig.add_shape(
                type="line",
                x0=-0.5, x1=4.5,  # 実測値の範囲に合わせる
                y0=threshold, y1=threshold,
                line=dict(color="red", width=2, dash="dash"),
                name="閾値"
            )

            # レイアウトの設定
            sigmoid_fig.update_layout(
                title=f"{c_target}の実測値 vs 予測確率 (SVM)",
                #dragmode="pan",
                xaxis=dict(title="実測値", tickvals=[0, 1, 3, 4], ticktext=["X", "O", "X", "O"], range=[-0.5, 4.5], fixedrange=False),
                yaxis=dict(title="予測値", range=[-0.1, 1.1], fixedrange=False)
            )
        
            cv_figures.append(sigmoid_fig)
            
            # OOFデータ出力用
            df_oof = pd.DataFrame(index=data.index, columns=["fold", "y_true", "y_pred"])
            kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=0)
            fold_colors = generate_fold_colors(n_fold)
            all_y_test, all_y_pred, all_y_proba = [], [], []
            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
                # 最適化済みパラメータで SVM を再構築
                pipe_cv = Pipeline([
                    ("prep", Preprocessor(impute_method=impute_method, scale=True)),
                    ("svm", SVC(
                        C=svm_opt.named_steps["svm"].C,
                        gamma=svm_opt.named_steps["svm"].gamma,
                        class_weight=svm_opt.named_steps["svm"].class_weight,
                        kernel="rbf",
                        probability=True,
                        random_state=0
                    ))
                ])
            
                pipe_cv.fit(X_train, y_train)
                cv_pred_proba = pipe_cv.predict_proba(X_test)[:, 1]  # positive class の確率
                cv_pred_binary = (cv_pred_proba >= threshold).astype(int)
                
                y_labels = ["O" if val == 1 else "X" for val in y_test]
                
                hovertemplate = (
                    f"<b>Fold {fold + 1}</b>"
                    "<br><b>実測値:</b> %{customdata}<br><b>予測陽性確率:</b> %{y}<extra></extra>"
                )
                
                sigmoid_fig.add_trace(go.Scatter(
                    x=3 + y_test + np.random.uniform(-jitter_width, jitter_width, size=len(y_test)),
                    y=cv_pred_proba,
                    customdata=y_labels,
                    mode="markers",
                    name=f"Fold {fold + 1}",
                    marker=dict(color=fold_colors[fold]),
                    hovertemplate=hovertemplate
                ))
                
                all_y_test.extend(y_test)
                all_y_pred.extend(cv_pred_binary)
                all_y_proba.extend(cv_pred_proba)   
                
                # fold の結果を OOF に追加
                df_oof.loc[dropped_data.index[test_idx], "fold"] = fold + 1
                df_oof.loc[dropped_data.index[test_idx], "y_true"] = y_test.astype(int)
                df_oof.loc[dropped_data.index[test_idx], "y_pred"] = cv_pred_proba.astype(float)
                    
            if "dynamic-tabs" in triggered_id:
                # DataFrame 化して export 用に保存
                df_oof["行数"] = df_oof.index + 1
                df_oof = df_oof[["行数", "fold", "y_true", "y_pred"]]
                cv_export_data.append(df_oof.to_dict("records"))
            else:
                cv_export_data.append(no_update)
                
            # サブプロット作成
            cm_figure = sp.make_subplots(
                            rows=2, cols=1,
                            subplot_titles=[f"{c_target}の混同行列(検証)", "評価指標"],
                            row_heights=[0.7, 0.3],
                            specs=[[{"type": "heatmap"}], [{"type": "table"}]]
                        )

            # 混同行列
            labels = svm.classes_
            cm = confusion_matrix(all_y_test, all_y_pred, labels=labels).T
            class_labels = ["X" if cl == 0 else "O" for cl in labels]  # クラスラベル ("X" と "O")
            
            cm_figure.add_trace(
                go.Heatmap(
                    z=cm,
                    x=class_labels,
                    y=class_labels,
                    text=cm,
                    texttemplate="%{text}",
                    colorscale=[[0, 'white'], [1, 'red']],
                    showscale=False,
                    hovertemplate=(
                        "実測値: %{x}<br>"
                        "予測値: %{y}<br>"
                        "値 (カウント): %{z}<extra></extra>"
                    )
                ),
                row=1, col=1
            )

            # --- 学習データの指標 ---
            train_metrics = {
                "Accuracy": accuracy_score(y, y_pred_binary),
                "Precision": precision_score(y, y_pred_binary),
                "Recall": recall_score(y, y_pred_binary),
                "F1 Score": f1_score(y, y_pred_binary)
            }
            
            # --- 検証データの指標 ---
            valid_metrics = {
                "Accuracy": accuracy_score(all_y_test, all_y_pred),
                "Precision": precision_score(all_y_test, all_y_pred),
                "Recall": recall_score(all_y_test, all_y_pred),
                "F1 Score": f1_score(all_y_test, all_y_pred)
            }
            
            # --- テーブル用データ作成 ---
            metric_names = [
                "Accuracy (正しく分類できた割合)",
                "Precision (陽性と分類して実際に陽性だった割合)",
                "Recall (実際の陽性を陽性と分類できた割合)",
                "F1 Score (PrecisionとRecallの両立度合い)"
            ]
            
            metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score"]
            
            metric_values = [
                metric_names,
                [train_metrics[k] for k in metric_keys],   # 学習
                [valid_metrics[k] for k in metric_keys]    # 検証
            ]
            
            # --- テーブル追加 ---
            cm_figure.add_trace(
                go.Table(
                    columnwidth=[260, 70, 70],
                    header=dict(
                        values=["指標", "学習", "検証"],
                        fill_color="lightgray",
                        align="center"
                    ),
                    cells=dict(
                        values=metric_values,
                        align="center",
                        format=[None, ".4f", ".4f"]
                    )
                ),
                row=2, col=1
            )
            
            r2 = matthews_corrcoef(all_y_test, all_y_pred)
            cm_figure.update_layout(
                title=f"{c_target}の分類結果 (SVM)  [一致度: {r2:.4f}]",
                xaxis=dict(
                    title="実測値",
                    tickmode="array",
                    tickvals=class_labels,
                    ticktext=class_labels,  
                    scaleanchor="y",
                    constrain="domain"
                ),
                yaxis=dict(
                    title="予測値",
                    tickmode="array",
                    tickvals=class_labels,
                    ticktext=class_labels
                ),
                dragmode=False
            )
            
            scatter_figures.append(cm_figure)

            # 結果メッセージ
            mae = mean_absolute_error(all_y_test, all_y_proba) * 100
            saved_cv_scores.append({"target": c_target, "cv_score": r2})
            result_message = html.Div([
                html.P(f"検証データへの一致度: {r2:.5f},　平均的な誤差: {mae:.3f}%", style={"color": "#3498db", "fontWeight": "bold"})
            ])
            result_messages.append(result_message)
            
            model_data.append({"name": c_target, "model": base64.b64encode(pickle.dumps(svm_opt)).decode('utf-8')})
                
        return [saved_cv_scores, model_data, result_messages, scatter_figures, importance_figures, cv_figures, cv_export_data]

    except Exception as e:
        # エラー時の処理
        return [
            [{"cv_score": None}] * len(total_variables),
            [{"model": None}] * len(total_variables),
            [html.Div(f"解析中にエラーが発生しました: {str(e)}", style={"color": "red"})] * len(total_variables),
            [default_fig("実測値 vs 予測値", "")] * len(total_variables),
            [default_fig("特徴量の重要度", "")] * len(total_variables),
            [default_fig("CV 実測値 vs 予測値", "")] * len(total_variables),
            [[]] * len(total_variables)
        ]

# コールバック: 説明変数ごとにスライダー/ドロップダウン作成
@app.callback(
    Output("input-fields", "children"),
    Input("analyze-button", "n_clicks"),
    [State("explanatory-variables", "value"),
     State("store-num-data", "data"),
     State("store-original-categories", "data"),
     State("store-rare-categories", "data")]
)
def generate_sliders(n_clicks, explanatory_variables, stored_data, original_categories, rare_category_map):
    if not explanatory_variables or stored_data is None:
        return html.Div("説明変数が選択されていません。", style={"color": "red"})

    data = pd.DataFrame(stored_data["data"])
    fields = []
    grouped = group_encoded_columns(explanatory_variables)  #グループ化
    
    processed_categories = {}

    for col, cats in original_categories["data"].items():
        rare_cats = rare_category_map["data"].get(col, [])
    
        if len(rare_cats) >= 2:
            # レアカテゴリを除外して「その他」を追加
            new_cats = [c for c in cats if c not in rare_cats]
            new_cats.append("その他")
            processed_categories[col] = new_cats
        else:
            # 統合されていないので元のカテゴリのまま
            processed_categories[col] = cats

    for base, cols in grouped.items():
        if base in processed_categories:
            # --- カテゴリ変数（One-hotエンコードされたもの） ---
            categories = processed_categories[base]
            options = [{"label": cat, "value": cat} for cat in categories]

            # ドロップダウン（ユーザーに見せる）
            fields.append(
                html.Div([
                    html.Label(f"{base}:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id={"type": "category-dropdown", "index": base},
                        options=options,
                        value=categories[0],
                        clearable=False,
                        style={"width": "70%"}
                    )
                ], style={"marginBottom": "20px"})
            )

            # 元のスライダー（制御用に残すが非表示）
            for col in cols:
                min_val = data[col].min() if not data[col].isnull().all() else 0
                max_val = data[col].max() if not data[col].isnull().all() else 1
                fields.append(
                    html.Div(
                        dcc.Slider(
                            id={"type": "range-slider", "index": col},
                            min=min_val,
                            max=max_val,
                            value=float(min_val),
                            step=None,  # 中間値を選べないようにする
                            marks={float(min_val): "X", float(max_val): "O"},
                        ),
                        style={"width": "40%", "display": "none"}  # 非表示にする
                    )
                )
        else:
            # --- 数値変数 ---
            col = cols[0]
            min_val = data[col].min() if not data[col].isnull().all() else 0
            max_val = data[col].max() if not data[col].isnull().all() else 1

            fields.append(
                html.Div([
                    html.Label(f"{col}:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                    html.Div([
                        dcc.Input(
                            id={"type": "min-input", "index": col},
                            type="number",
                            value=float(min_val),
                            style={"width": "15%", "marginRight": "10px", "fontSize": "12px"}
                        ),
                        html.Div(
                            dcc.Slider(
                                id={"type": "range-slider", "index": col},
                                min=float(min_val),
                                max=float(max_val),
                                value=float(min_val),
                                marks=None,
                                tooltip={"always_visible": True, "placement": "bottom"}
                            ),
                            style={"width": "65%", "marginRight": "10px"}
                        ),
                        dcc.Input(
                            id={"type": "max-input", "index": col},
                            type="number",
                            value=float(max_val),
                            style={"width": "15%", "marginRight": "10px", "fontSize": "12px"}
                        )
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})
                ], style={"marginBottom": "20px"})
            )

    return fields

# コールバック: スライダーの範囲更新
@app.callback(
    [Output({"type": "range-slider", "index": MATCH}, "min"),
     Output({"type": "range-slider", "index": MATCH}, "max")],
    [Input({"type": "min-input", "index": MATCH}, "value"),
     Input({"type": "max-input", "index": MATCH}, "value")]
)
def update_slider_range(min_value, max_value):
    if min_value is None or max_value is None or min_value >= max_value:
        return no_update, no_update
    return min_value, max_value

# コールバック: ドロップダウンに応じて非表示スライダー更新
@app.callback(
    Output({"type": "range-slider", "index": ALL}, "value"),
    Input({"type": "category-dropdown", "index": ALL}, "value"),
    [State({"type": "category-dropdown", "index": ALL}, "id"),
     State({"type": "range-slider", "index": ALL}, "id"),
     State("store-num-data", "data"),
     State("explanatory-variables", "value")]
)
def update_hidden_sliders(selected_categories, dropdown_ids, slider_ids, stored_data, explanatory_variables):

    data = pd.DataFrame(stored_data["data"])
    grouped = group_encoded_columns(explanatory_variables)

    # UI で選ばれたカテゴリを辞書化
    selected_map = {
        dd_id["index"]: sel
        for sel, dd_id in zip(selected_categories, dropdown_ids)
    }

    results = []

    # スライダーの順番でループ
    for sid in slider_ids:
        col = sid["index"]

        # --- 数値スライダーは no_update ---
        if "[" not in col:
            results.append(no_update)
            continue

        # --- カテゴリスライダー ---
        base = col.split("[")[0]
        cat = col[len(base)+1:-1]

        selected = selected_map.get(base, None)

        # sum-to-zero / one-hot 共通で min/max を使う
        min_val = float(data[col].min()) if not data[col].isnull().all() else 0
        max_val = float(data[col].max()) if not data[col].isnull().all() else 1

        # cols に存在しないカテゴリ = drop-first の基準カテゴリ
        if selected is not None and selected not in [c[len(base)+1:-1] for c in grouped[base]]:
            results.append(min_val)
        else:
            results.append(max_val if cat == selected else min_val)

    return results

# コールバック: 予測値計算
@app.callback(
    [Output("prediction-results", "children"),
     Output("store-scalers", "data")],
    [Input("tabs", "value"),
     Input({"type": "range-slider", "index": ALL}, "value")],
    [State("explanatory-variables", "value"),
     State("target-variables", "value"),
     State("class-variables", "value"),
     State("store-num-data", "data"),
     State("store-class-data", "data"),
     State({"type": "threshold-slider", "index": ALL}, "value"),
     State({"type": "saved-cv-scores", "index": ALL}, "data"),
     State({"type": "stored-model", "index": ALL}, "data"),
     State("impute-mode", "value")],
    prevent_initial_call=True
)
def update_predictions(selected_tab, slider_values, explanatory_variables, target_variables, class_variables, 
                       stored_data, stored_class_data, threshold_values, saved_cv_scores, model_data, impute_method):
    
    if selected_tab != "prediction-tab":
        return no_update, no_update  # 別タブなら何もしない
    
    missing_models = []
    
    if not slider_values or model_data is None:
        return html.Div("解析終了後、スライダーを操作して結果を表示", style={"color": "red"}), no_update

    else:
        # スライダー値を使った入力データを生成
        input_data = pd.DataFrame([slider_values], columns=explanatory_variables)
        data = pd.DataFrame(stored_data["data"])
        class_data = pd.concat([data[explanatory_variables], pd.DataFrame(stored_class_data["data"])], axis=1)

        # サブプロット作成（目的変数の数だけ列を作成）
        total_variables = target_variables + class_variables
        fig = sp.make_subplots(
            rows=1, cols=len(total_variables),
            shared_yaxes=False,  # y軸はそれぞれ独立
        )
        
        scaler_list = []

        for i, (target, score) in enumerate(zip(target_variables, saved_cv_scores)):
            # 回帰モデルを構築して予測値を計算
            dropped_data = data.dropna(subset=[target])
            y = dropped_data[target].values
            
            gpr_opt = load_model(model_data, target)
            if gpr_opt is None:
                missing_models.append(target)
                continue

            # Pipeline の scaler を取り出す 
            prep = gpr_opt.named_steps["prep"] 
            scaler_list.append({
                "center": prep.scaler.center_.tolist(),
                "scale": prep.scaler.scale_.tolist()
            })
            
            prediction, uncertainty = gpr_opt.predict(input_data, return_std=True)
            uncertainty = uncertainty * 1.96

            # 箱ひげ図をサブプロットに追加
            fig.add_trace(
                go.Box(
                    y=y,
                    name=target,
                    boxpoints=False,
                    fillcolor="rgba(0, 0, 255, 0.2)",
                    line=dict(color="rgba(0, 0, 255, 0.5)")
                ),
                row=1, col=i + 1
            )

            # 予測値と不確実性をプロット
            fig.add_trace(
                go.Scatter(
                    x=[target],
                    y=[prediction[0]],
                    mode="markers+text",
                    text=[f"{prediction[0]:.2f}±{uncertainty[0]:.2f}"],
                    textfont=dict(color="red", size=14),
                    textposition="top center",
                    marker=dict(color="red", size=10),
                    name="予測値(95%信頼区間)",
                    error_y=dict(
                        type="data",
                        array=[uncertainty[0]],
                        visible=True,
                        thickness=0.3,
                        width=4,
                    )
                ),
                row=1, col=i + 1
            )

            # x軸タイトルにCVのr²スコアを表示
            fig.update_xaxes(
                title=dict(
                    text=f"一致度: {score['cv_score']:.3f}",
                    font=dict(size=12, color="darkblue")
                ),
                row=1, col=i + 1
            )

        for j, (c_target, score) in enumerate(zip(class_variables, saved_cv_scores[len(target_variables):])):
            # 分類モデルを構築して陽性確率を計算
            dropped_data = class_data.dropna(subset=[c_target])
            y = dropped_data[c_target].map({"O": 1, "X": 0}).values.astype(int)

            svm_opt = load_model(model_data, c_target)
            if svm_opt is None:
                missing_models.append(c_target)
                continue

            # Pipeline の scaler を取り出す 
            prep = svm_opt.named_steps["prep"] 
            scaler_list.append({ 
                "center": prep.scaler.center_.tolist(), 
                "scale": prep.scaler.scale_.tolist() 
            })
        
            prediction_prob = svm_opt.predict_proba(input_data)[0, 1]  # 陽性確率

            # 確率をサブプロットに追加（単純な棒グラフ）
            threshold_idx = len(target_variables) + j
            if threshold_idx < len(threshold_values)and threshold_values[threshold_idx] is not None:
                threshold = threshold_values[threshold_idx]
            else:
                threshold = 0.5
            
            if prediction_prob >= threshold:
                bar_color = "rgba(255, 0, 0, 0.3)"
                text_color = "rgb(255, 0, 0)"
                text_OX = "O"
            else:
                bar_color = "rgba(0, 0, 255, 0.3)",
                text_color = "rgb(0, 0, 255)",
                text_OX = "X"
                
            fig.add_trace(
                go.Bar(
                    x=[c_target],
                    y=[prediction_prob],
                    text=[f"{prediction_prob:.2%} → {text_OX}"],
                    textfont=dict(color=text_color, size=14),
                    textposition="outside",
                    marker=dict(color=bar_color),
                    name="陽性確率"
                ),
                row=1, col=len(target_variables) + j + 1
            )

            # x軸タイトルにCVのスコアを表示
            fig.update_xaxes(
                title=dict(
                    text=f"一致度: {score['cv_score']:.3f}",
                    font=dict(size=12, color="darkblue")
                ),
                row=1, col=len(target_variables) + j + 1
            )
            fig.update_yaxes(
                range=[0, 1.1],  # y 軸の範囲を [0, 1.1] に設定
                row=1, col=len(target_variables) + j + 1
            )

        # レイアウト設定
        fig.update_layout(
            title="目的変数の分布と予測値 (GPR/SVM)",
            showlegend=False,
            plot_bgcolor="#f9f9f9",
        )
        
        # 一括予測アップロード部分
        bulk_predict = html.Div(
            style={"marginTop": "30px"},
            children=[
                html.Label("一括予測:", style={"marginRight": "10px", "fontWeight": "bold"}),
        
                dcc.Upload(
                    id="bulk-predict-upload",
                    children=html.Button("アップロード", style={"marginLeft": "10px"}),
                    multiple=False,
                    accept=".csv",
                    style={"display": "inline-block"}
                ),
        
                html.Span(id="bulk-predict-status",
                          style={"marginLeft": "15px", "fontWeight": "bold"}),
        
                dcc.Download(id="bulk-predict-download")
            ]
        )
        
        if missing_models:
            missing_text = f"モデルが見つかりません: {', '.join(missing_models)}"
            return html.Div(missing_text, style={"color": "red"}), no_update
        else:
            return html.Div(
                        children=[
                            dcc.Graph(figure=fig),
                            bulk_predict
                        ]
                    ), scaler_list
    '''
    except Exception as e:
        return html.Div(f"エラーが発生しました: {str(e)}", style={"color": "red"}), no_update
        #return html.Div("解析終了後、スライダーを操作して結果を表示", style={"color": "red"}), no_update
    '''
@app.callback(
    [Output("bulk-predict-status", "children"),
     Output("bulk-predict-status", "style"),
     Output("bulk-predict-download", "data")],
    Input("bulk-predict-upload", "contents"),
    [State("bulk-predict-upload", "filename"),
    State("explanatory-variables", "value"),
    State("target-variables", "value"),
    State("class-variables", "value"),
    State({"type": "dynamic-dropdown", "col": ALL}, "value"),
    State({"type": "dynamic-dropdown", "col": ALL}, "id"),
    State("impute-mode", "value"),
    State({"type": "threshold-slider", "index": ALL}, "value"),
    State({"type": "saved-cv-scores", "index": ALL}, "data"),
    State("store-num-data", "data"),
    State("store-class-data", "data"),
    State("store-label-data", "data"),
    State({"type": "stored-model", "index": ALL}, "data"),
    State("store-rare-categories", "data")],
    prevent_initial_call=True
)
def bulk_predict(contents, filename, explanatory_vars, target_variables, class_variables,
                 dropdown_values, dropdown_ids, impute_mode, threshold_values, saved_cv_scores,
                 stored_num_data, stored_class_data, label_data, model_data, rare_category_store):

    if contents is None:
        return ["", {}, None]

    # -------------------------
    # ① CSV 読み込み
    # -------------------------
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    result = chardet.detect(decoded)
    encoding = result["encoding"] if result["encoding"] else "utf-8"

    try:
        df_new = pd.read_csv(io.StringIO(decoded.decode(encoding)))
    except:
        df_new = pd.read_csv(io.StringIO(decoded.decode("cp932")))

    # 「%」付き表記を数値に変換（例: 10% -> 0.1）
    df_new = convert_percent_columns(df_new)

    # -------------------------
    # ② 説明変数チェック
    # -------------------------
    selected_explanatory = sorted({col.split("[")[0] for col in explanatory_vars})
    missing = [col for col in selected_explanatory if col not in df_new.columns]
    if missing:
        return [f"不足している変数: {', '.join(missing)}", {"color": "red"}, None]
    
    if impute_mode == "none":
        missing_cols = [col for col in selected_explanatory if df_new[col].isna().any()]
        if missing_cols:
            return [f"欠損がある変数: {', '.join(missing_cols)}", {"color": "red"}, None]

    # -------------------------
    # ③ 説明変数・目的変数のエンコード
    # -------------------------

    rare_category_map = {}
    if rare_category_store and "data" in rare_category_store:
        rare_category_map = rare_category_store["data"]

    df_proc = df_new.copy()
    
    # レアカテゴリ置換（train と同じルール）
    for col, rare_cats in rare_category_map.items():
    
        # test 側に列が存在しない場合はスキップ
        if col not in df_proc.columns:
            continue
    
        # rare_cats が空なら統合なし
        if not rare_cats:
            continue
    
        df_proc[col] = df_proc[col].apply(
            lambda x: "その他" if x in rare_cats else x
        )

    num_data_base = pd.DataFrame()
    
    # ラベルデータ（目的変数候補）
    df_category = pd.DataFrame(label_data["data"])
    df_category.columns = label_data["columns"]
    
    # --- 元の構造を維持したループ ---    
    for col in selected_explanatory:
    
        # CSVに列が無ければスキップ
        if col not in df_proc.columns:
            continue
    
        # 数値列
        if pd.api.types.is_numeric_dtype(df_proc[col]):
            num_data_base[col] = df_proc[col]
            continue
    
        # カテゴリ列
        dummies = pd.get_dummies(
            df_proc[col],
            prefix="",
            prefix_sep="",
            drop_first=False
        )
    
        dummies.columns = [f"{col}[{c}]" for c in dummies.columns]
        dummies.loc[df_proc[col].isna(), :] = np.nan
        dummies = dummies.astype(float)
    
        num_data_base = pd.concat([num_data_base, dummies], axis=1)
    
    
    # -------------------------
    # ④ 列順を揃える
    # -------------------------
    num_data = pd.DataFrame()
    for col in explanatory_vars:
        if col in num_data_base.columns:
            num_data[col] = num_data_base[col]
        else:
            num_data[col] = 0.0
    
    df_encoded = num_data.copy()

    # -------------------------
    # ⑤ 予測実行
    # -------------------------
    df_result = df_new.copy()

    # --- 回帰モデル ---
    for i, (target, score) in enumerate(zip(target_variables, saved_cv_scores)):

        gpr_opt = load_model(model_data, target)
        if gpr_opt is None:
            df_result[target] = np.nan
            continue

        # Pipeline が自動で transform → predict
        pred_mean, pred_std = gpr_opt.predict(
            df_encoded,
            return_std=True
        )
        
        df_result[f"{target}_mean"] = pred_mean
        df_result[f"{target}_std"] = pred_std

    # --- 分類モデル ---
    for j, (c_target, score) in enumerate(zip(class_variables, saved_cv_scores[len(target_variables):])):

        svm_opt = load_model(model_data, c_target)
        if svm_opt is None:
            df_result[c_target] = np.nan
            continue

        threshold = threshold_values[len(target_variables) + j]

        prob = svm_opt.predict_proba(df_encoded)[:, 1]
        df_result[c_target] = np.where(prob >= threshold, "O", "X")
        df_result[f"{c_target}_proba"] = prob

    # -------------------------
    # ⑦ CSV 出力
    # -------------------------
    try:
        csv_bytes = df_result.to_csv(index=False, encoding="cp932").encode("cp932")
    except UnicodeEncodeError:
        csv_bytes = df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    b64 = base64.b64encode(csv_bytes).decode("ascii")
    base_name, _ = os.path.splitext(filename)
    new_filename = f"{base_name}_GPR.csv"

    return (
        "予測完了",
        {"color": "green"},
        dict(content=b64, filename=new_filename, base64=True)
    )
    
# コールバック: 推奨条件生成用設定画面
@app.callback(
    [Output("target-variable-table", "children"),
     Output("class-variable-table", "children"),
     Output("explanatory-variable-table", "rowData")],
    Input("analyze-button", "n_clicks"),
    [State("explanatory-variables", "value"),
     State("target-variables", "value"),
     State("class-variables", "value")]
)
def update_table(n_clicks, explanatory_values, target_values, class_values):
    
    try:
        if not target_values:
            target_children = []
        else:
            target_children = [
                html.H5("回帰の目標範囲設定", style={"color": "#5C7B99", "fontWeight": "bold"}),
                *[
                    html.Div([
                        html.Label(f"{target}", style={"flex": "8", "margin-right": "10px"}),
                        dcc.Input(id={"type": "input-lower", "index": target}, type="number", placeholder="下限値", style={"flex": "2", "margin-right": "10px"}),
                        dcc.Input(id={"type": "input-upper", "index": target}, type="number", placeholder="上限値", style={"flex": "2"}),
                    ], style={"display": "flex", "border": "1px solid #E0E0E0", "padding": "10px"})
                    for target in target_values
                ],
                html.Div(id="error-target", style={"color": "red", "margin-top": "10px", "textAlign": "left"})
            ]
            
        if not class_values:
            class_children = []
        else:
            class_children = [
                html.H5("分類の目標設定", style={"color": "#5C7B99", "fontWeight": "bold", "margin-top": "20px"}),
                *[
                    html.Div([
                        html.Label(f"{class_var}", style={"flex": "8", "margin-right": "10px"}),
                        dcc.RadioItems(
                            id={"type": "category", "index": class_var},
                            options=[{"label": "O", "value": "O"}, {"label": "X", "value": "X"}, {"label": "使用しない", "value": "-"}],
                            value="O",
                            inline=True,
                            style={"flex": "4"},
                            inputStyle={"marginLeft": "20px", "marginRight": "5px"}
                        ),
                    ], style={"display": "flex", "border": "1px solid #E0E0E0", "padding": "10px"})
                    for class_var in class_values
                ]
            ]

        explanatory_data = [{"説明変数": var, "水準値": ""} for var in explanatory_values] if explanatory_values else []

        return target_children, class_children, explanatory_data
    except Exception:
        return [], [], []

# 推奨点生成・計算
@app.callback(
    [Output('output-results', 'children'),
     Output('prediction-container', 'children'),
     Output('map-container', 'children')],
    Input('recommend-button', 'n_clicks'),
    [State("store-label-data", "data"),
    State('explanatory-variable-table', 'rowData'),
    State('target-variable-table', 'children'),
    State('class-variable-table', 'children'),
    State('constraints-table', 'data'),
    State('num-exp', 'value'),
    State('opt-type', 'value'),
    State('store-num-data', 'data'),
    State('store-class-data', 'data'),
    State({"type": "input-lower", "index": ALL}, "value"),
    State({"type": "input-upper", "index": ALL}, "value"),
    State({"type": "category", "index": ALL}, "value"),
    State({"type": "stored-model", "index": ALL}, "data"),
    State({"type": "threshold-slider", "index": ALL}, "value"),
    State("label-variables", "value")],
    prevent_initial_call=True
)
def analyze_data(n_clicks, label_data, explanatory_table, target_children, class_children, constraints, num_exp, opt_type,
                 store_data, class_data, lowers, uppers, categories, model_data, threshold_values, label_variable):
    if explanatory_table is None or not num_exp or not opt_type:
        return "すべての必要な情報を入力してください。", None, None
    elif not model_data or all(m is None for m in model_data):
        return "モデルが見つかりません。解析終了後に実行してください。", None, None

    # 入力データの水準を読み込む
    variable_levels = {}
    
    def parse_levels(levels_str):
        levels = []
        items = levels_str.split(",")
        i = 0
        while i < len(items):
            item = items[i].strip()
            if item in {"...", "…"}:
                if i < 2 or i >= len(items) - 1:
                    raise ValueError("省略記号 '...' の前後に最低2つの値が必要です")
                start = float(items[i - 1])
                prev = float(items[i - 2])
                end = float(items[i + 1])
                step = start - prev
                if step == 0:
                    raise ValueError("ステップ幅が0です")
    
                # 展開
                current = start + step
                while (step > 0 and current < end) or (step < 0 and current > end):
                    levels.append(round(current, 10))
                    current += step
    
                # 最後の値が含まれていなければ追加
                if round(levels[-1], 10) != round(end, 10):
                    levels.append(round(end, 10))
    
                i += 2
            else:
                try:
                    val = float(item)
                    levels.append(val)
                except ValueError:
                    raise ValueError(f"'{item}' は数値として認識できません")
                i += 1
        return levels

    for row in explanatory_table:
        var = row.get("説明変数")
        levels_str = row.get("水準値")
    
        if not var or not levels_str:
            return f"{var} の水準値が未入力です。(セルが編集中の場合は、Enterキーなどで値を確定してください。)", None, None
        try:
            variable_levels[var] = parse_levels(levels_str)
        except Exception as e:
            return f"{var} の水準値に誤りがあります：{str(e)}", None, None

    # 候補点が多すぎる場合はエラーを出す
    total_candidates = 1
    for lvl in variable_levels.values():
        total_candidates *= len(lvl)
    if total_candidates > 1e+6:
        return f"候補点が多すぎます(100万点以上)。水準数を減らしてください。　現在の候補点: {total_candidates:,}点", None, None

    # 全ての実験点を生成
    all_combinations = list(product(*variable_levels.values()))
    
    # 制約をかける
    def apply_constraints(all_exps, constraints):
        filtered_exps = []
        for exp in all_exps:
            meets_constraints = True
            for constraint in constraints:
                selected_vars = constraint["変数"].split(", ")
                sum_range = constraint["合計値制約"].split("～")
                select_range = constraint["選択数制約"].split("～")

                # 合計値制約チェック
                if sum_range != ["なし"]:
                    sum_min = float("-inf") if sum_range[0] == "" else float(sum_range[0])
                    sum_max = float("inf") if sum_range[1] == "" else float(sum_range[1])
                    
                    try:
                        selected_sum = sum(float(exp[i]) for i, var in enumerate(variable_levels.keys()) if var in selected_vars)
                    except ValueError:
                        return "制約をかける変数の水準に数値以外の値が含まれています。", None, None
                        
                    if not (sum_min <= selected_sum <= sum_max):
                        meets_constraints = False
                        break

                # 選択数制約チェック
                if select_range != ["なし"]:
                    select_min, select_max = map(int, select_range)
                    
                    try:
                        selected_count = sum(1 for i, var in enumerate(variable_levels.keys()) if var in selected_vars and float(exp[i]) != 0)
                    except ValueError:
                        return "制約をかける変数の水準に数値以外の値が含まれています。", None, None
                    if not (select_min <= selected_count <= select_max):
                        meets_constraints = False
                        break

            if meets_constraints:
                filtered_exps.append(exp)

        return np.array(filtered_exps)

    all_exps = apply_constraints(all_combinations, constraints)
    
    # 制約後の実験点が残っていない/多すぎる場合
    if len(all_exps) == 0:
        return "制約を満たす実験点がありません。制約条件を見直してください。(選択数制約を使用しているときは、水準値に0が入っているか確認してください。)", None, None
    elif len(all_exps) < num_exp:
        return f"制約を満たす実験点が少なすぎます。水準数を増やすか制約条件を緩めてください。　現在の候補点: {len(all_exps)}点", None, None

    exp_df = pd.DataFrame(all_exps, columns=variable_levels.keys())

    # 回帰用変数リスト
    target_variables = []
    for div in target_children:
        if isinstance(div, dict) and div.get("type") == "Div":
            children = div.get("props", {}).get("children", [])
            if isinstance(children, list) and len(children) >= 1:
                label_component = children[0]
                if isinstance(label_component, dict) and label_component.get("type") == "Label":
                    var_name = label_component.get("props", {}).get("children")
                    if isinstance(var_name, str):
                        target_variables.append(var_name)
    
    # 分類用変数リスト
    class_variables = []
    for div in class_children:
        if isinstance(div, dict) and div.get("type") == "Div":
            children = div.get("props", {}).get("children", [])
            if isinstance(children, list) and len(children) >= 1:
                label_component = children[0]
                if isinstance(label_component, dict) and label_component.get("type") == "Label":
                    var_name = label_component.get("props", {}).get("children")
                    if isinstance(var_name, str):
                        class_variables.append(var_name)
    
    total_variables = target_variables + class_variables
    
    # 獲得関数計算
    data_df = pd.DataFrame(store_data["data"])
    class_df = pd.DataFrame(class_data["data"]) 

    def regression_score(model, X, lower=None, upper=None, opt_type="2"):
        mu, sigma = model.predict(X, return_std=True) 
        kappa = 0.5 * (int(opt_type) - 0.5) 
        eps = 1e-9 
        
        if lower is not None and upper is None: # 下限のみ → 最大化 
            PI = norm.cdf((mu - lower) / (sigma + eps)) 
            utility = mu + kappa * sigma 
        elif upper is not None and lower is None: # 上限のみ → 最小化 
            PI = norm.cdf((upper - mu) / (sigma + eps)) 
            utility = -(mu - kappa * sigma) 
        elif lower is not None and upper is not None: # 範囲指定 
            z_u = (upper - mu) / (sigma + eps) 
            z_l = (lower - mu) / (sigma + eps) 
            PI = norm.cdf(z_u) - norm.cdf(z_l) 
            m = 0.5 * (lower + upper) 
            utility = -np.abs(mu - m) + kappa * sigma
        else: # 制約なし 
            PI = np.ones_like(mu) 
            utility = mu 
            
        # rank-based normalization 
        r = rankdata(utility, method="average") 
        utility_norm = (r - 1) / (len(r) - 1) 
        return PI * utility_norm

    def classification_score(model, X, goal, threshold):
        prob = model.predict_proba(X)[:, 1]
        
        score = 0.5* (1 + erf(3 * (prob - threshold)/(0.5 - abs(threshold - 0.5))))
        if goal == "O":
            return score
        elif goal == "X":
            return 1 - score
        return np.zeros_like(prob)

    reg_scores = []
    cls_scores = []

    # 全候補点に対する回帰モデルのスコア
    for i, var in enumerate(target_variables):
        model = load_model(model_data, var)        
        lower = lowers[i] if i < len(lowers) and lowers[i] is not None else None
        upper = uppers[i] if i < len(uppers) and uppers[i] is not None else None
        if model:
            score = regression_score(model, exp_df, lower, upper, opt_type)
            reg_scores.append(score)

    # 全候補点に対する分類モデルのスコア
    for i, var in enumerate(class_variables):
        model = load_model(model_data, var)
        category = categories[i] if i < len(categories) else "-"
        if model:
            var_idx = total_variables.index(var)
            if var_idx < len(threshold_values) and threshold_values[var_idx] is not None:
                threshold = threshold_values[var_idx]
            else:
                threshold = 0.5
                
            score = classification_score(model, exp_df, category, threshold)
            cls_scores.append(score)
    
    def regression_weight(mean, data_range, lower=None, upper=None):
        if lower is not None and upper is not None: # 範囲
                dist = max(0, lower - mean, mean - upper)
        elif lower is not None: # 最大化
            dist = max(0, lower - mean)
        elif upper is not None: # 最小化
            dist = max(0, mean - upper)
        else:
            return 0.0
    
        t = dist / data_range
        return 0.2 + 0.8 * (1 + t)**3

    def classification_weight(x):
        # x = 目標カテゴリの出現率（0〜1）
        return 4 * 16**(-x)

    # 回帰の変数ごとの重み
    reg_weights = []
    if reg_scores:
        for i, score in enumerate(reg_scores):
            var = target_variables[i]
            mean = data_df[var].mean()
            v_min = data_df[var].min()
            v_max = data_df[var].max()
            data_range = v_max - v_min
            
            # 全データが同じ値の場合などのガード処理
            if data_range < 1e-10:
                # データの絶対値の10%程度を仮のレンジとする
                data_range = abs(data_df[var].mean()) * 0.1 + 1e-10
                
            lower = lowers[i] if i < len(lowers) and lowers[i] is not None else None
            upper = uppers[i] if i < len(uppers) and uppers[i] is not None else None
    
            weight = regression_weight(mean, data_range, lower, upper)
            reg_weights.append(weight)
    else:
        reg_weights = []
    
    # 分類の変数ごとの重み
    cls_weights = []
    if cls_scores:
        for i, score in enumerate(cls_scores):
            var = class_variables[i]
            category = categories[i] if i < len(categories) else "-"
            if category not in ["O", "X"]:
                continue
    
            valid_mask = class_df[var].isin(["O", "X"])
            if valid_mask.sum() == 0:
                continue
    
            x = (class_df.loc[valid_mask, var] == category).mean()
            weight = classification_weight(x)
            cls_weights.append(weight)
    else:
        cls_weights = []

    # --- スコア合成 ---
    all_scores = reg_scores + cls_scores   # 各要素は shape (n_candidates,)
    normalized_scores = []
    
    for s in all_scores:
        s = np.asarray(s)
        s_min = np.nanmin(s)
        s_max = np.nanmax(s)
    
        if np.isclose(s_max, s_min):
            # 全部同じ → 影響させない
            s_norm = np.zeros_like(s)
        else:
            s_norm = (s - s_min) / (s_max - s_min + 1e-12)
    
        normalized_scores.append(s_norm)

    multi_scores = np.vstack(all_scores).T
    weights = np.array(reg_weights + cls_weights)
    weights = weights / (np.mean(weights))
    final_score = np.sum(weights * multi_scores, axis=1)
    
    def mixed_distance(X1, X2, groups, variable_levels, exp_df):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        D = np.zeros((n1, n2))
        var_list = list(variable_levels.keys())
    
        global_ranges = {}
        for col in var_list:
            col_data = exp_df[col].astype(float).values
            r = np.nanmax(col_data) - np.nanmin(col_data)
            global_ranges[col] = r if r > 1e-12 else 1.0
    
        for base, cols in groups.items():
            idx = [var_list.index(c) for c in cols]
            block1 = X1[:, idx]
            block2 = X2[:, idx]
    
            if len(cols) == 1:
                # 数値 → 正規化ユークリッド
                rng = global_ranges[cols[0]]
                # ブロードキャストで差分行列を作成
                d = np.abs(block1[:, None, 0] - block2[None, :, 0]) / rng
            else:
                # カテゴリ → ハミング
                valid1 = block1.sum(axis=1) > 0
                valid2 = block2.sum(axis=1) > 0
                label1 = np.where(valid1, np.argmax(block1, axis=1), -1)
                label2 = np.where(valid2, np.argmax(block2, axis=1), -1)
                d = (label1[:, None] != label2[None, :]).astype(float)
                missing = (label1[:, None] == -1) | (label2[None, :] == -1)
                d[missing] = 0.5
            D += d
        
        return D / max(len(groups), 1)
    
    def score_dist_selection(X_candidates, X_past, scores, num_points, opt_type):
        n_samples = len(X_candidates)
        groups = group_encoded_columns(variable_levels.keys())
    
        # 重みの設定
        mode_weights = {
            "1": (0.8, 0.1, 0.1),
            "2": (0.5, 0.3, 0.2),
            "4": (0.2, 0.5, 0.3),
        }
        score_w, novelty_w, diversity_w = mode_weights.get(str(opt_type), (0.5, 0.3, 0.2))
    
        # 1. スコアの正規化
        score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    
        # 2. 過去データとの距離（未知度）
        d_to_train = mixed_distance(X_candidates, X_past, groups, variable_levels, exp_df).min(axis=1)
        novelty_norm = (d_to_train - d_to_train.min()) / (d_to_train.max() - d_to_train.min() + 1e-12)
    
        # 3. 1点目の選択
        combined_first = score_w * score_norm + novelty_w * novelty_norm
        first_idx = np.argmax(combined_first)
        
        selected_indices = [first_idx]
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[first_idx] = False
    
        # 4. 候補点同士の距離行列
        dist_matrix = mixed_distance(X_candidates, X_candidates, groups, variable_levels, exp_df)
    
        
        min_dists_raw = dist_matrix[:, first_idx]  # 選択済み点からの最小距離を保持
    
        # 5. 逐次選択ループ
        while len(selected_indices) < num_points:
            d_min = min_dists_raw[remaining_mask].min()
            d_max = min_dists_raw[remaining_mask].max()
            
            diversity_norm = np.zeros_like(min_dists_raw)
            if d_max > d_min:
                diversity_norm[remaining_mask] = (min_dists_raw[remaining_mask] - d_min) / (d_max - d_min + 1e-12)
            
            # 複合スコアの計算
            combined_score = (
                score_w * score_norm +
                novelty_w * novelty_norm +
                diversity_w * diversity_norm
            )
    
            combined_score[~remaining_mask] = -np.inf
            best_idx = np.argmax(combined_score)
    
            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False
            
            # 生の最小距離を更新
            min_dists_raw = np.minimum(min_dists_raw, dist_matrix[:, best_idx])
    
        return selected_indices
    
    # 実験点を逐次的に選択
    # --- 1. 候補点の絞り込みと多様性選択の修正 ---
    top_k = min(len(final_score), 10000)  # スコア上位のインデックスを抽出
    top_score_idx_full = np.argsort(final_score)[-top_k:][::-1]  # スコア降順
    
    # 上位k点だけのDataFrameを作成（インデックスを0からにリセット）
    top_candidates_df = exp_df.iloc[top_score_idx_full].copy()
    top_candidates_df['temp_total_score'] = final_score[top_score_idx_full]
    
    X_for_dist = top_candidates_df[variable_levels.keys()].astype(float).values
    score_for_dist = top_candidates_df['temp_total_score'].values
    X_past = data_df[variable_levels.keys()].astype(float).values
    
    selected_rel_indices = score_dist_selection(X_for_dist, X_past, score_for_dist, num_exp, opt_type)
    
    # 最終的な推奨点を取得
    top_df = top_candidates_df.iloc[selected_rel_indices].copy()
    top_df = top_df.drop(columns=['temp_total_score'])

    # --- 2. 予測結果の計算 ---
    # 回帰目的変数
    for i, var in enumerate(target_variables):
        model = load_model(model_data, var)
        if model:
            mu, sigma = model.predict(top_df[variable_levels.keys()], return_std=True)
            top_df[f"{var}_mean"] = mu.flatten()
            top_df[f"{var}_std"] = sigma.flatten()
    
    # 分類目的変数
    for i, var in enumerate(class_variables):
        model = load_model(model_data, var)
        category = categories[i] if i < len(categories) else "-"
        if model and category in ["O", "X"]:
            prob = model.predict_proba(top_df[variable_levels.keys()])[:, 1]
            
            var_idx = total_variables.index(var) 
            threshold = threshold_values[var_idx] if (var_idx < len(threshold_values) and threshold_values[var_idx] is not None) else 0.5
            
            top_df[f"陽性確率_{var}"] = prob
            top_df[f"判定_{var}"] = np.where(prob >= threshold, "O", "X")

    # --- 3. グラフ描画ロジック ---
    total_vars = target_variables + class_variables
    fig = sp.make_subplots(rows=1, cols=len(total_vars), subplot_titles=total_vars)
    # 推奨点数に合わせて色を生成
    colors = plotly.colors.sample_colorscale("Viridis", [i / max(1, (len(top_df)-1)) for i in range(len(top_df))])
    
    for i, var in enumerate(target_variables):
        y_train = pd.DataFrame(store_data["data"])[var].dropna()
        mu = top_df[f"{var}_mean"].values
        sigma = top_df[f"{var}_std"].values
        
        # 表示範囲の計算
        y_min, y_max = y_train.min(), y_train.max()
        pred_min, pred_max = (mu - 2.2 * sigma).min(), (mu + 2.2 * sigma).max()
        y_lower_bound = min(y_min, pred_min)
        y_upper_bound = max(y_max, pred_max)

        # 既存データの箱ひげ
        fig.add_trace(go.Box(
            y=y_train, name="既存データ", boxpoints=False,
            fillcolor="rgba(0, 0, 255, 0.2)", line=dict(color="rgba(0, 0, 255, 0.5)"),
            showlegend=False
        ), row=1, col=i+1)
        
        # 目標範囲の背景色（緑）
        raw_lower = lowers[i] if (i < len(lowers) and lowers[i] not in [None, ""]) else None
        raw_upper = uppers[i] if (i < len(uppers) and uppers[i] not in [None, ""]) else None
        
        target_low = float(raw_lower) if raw_lower is not None else y_lower_bound - abs(y_lower_bound)*0.2
        target_high = float(raw_upper) if raw_upper is not None else y_upper_bound + abs(y_upper_bound)*0.2
        
        if raw_lower is not None or raw_upper is not None:
            fig.add_shape(
                type="rect", xref="x", yref="y", x0=-0.5, x1=len(top_df)+0.5,
                y0=target_low, y1=target_high, fillcolor="rgba(0, 255, 0, 0.2)",
                line=dict(width=0), row=1, col=i+1
            )

        # 推奨点のプロット
        for j in range(len(top_df)):
            fig.add_trace(go.Scatter(
                x=[f"推奨点_{j+1}"], y=[mu[j]], mode="markers",
                marker=dict(color=colors[j], size=10),
                error_y=dict(type="data", array=[1.96*sigma[j]], visible=True, thickness=0.3, width=4),
                showlegend=False
            ), row=1, col=i+1)

    # 分類グラフ
    for j, var in enumerate(class_variables):
        prob = top_df[f"陽性確率_{var}"].values
        label = top_df[f"判定_{var}"].values
        for i in range(len(top_df)):
            fig.add_trace(go.Bar(
                x=[f"推奨点_{i+1}"], y=[prob[i]],
                text=[f"{prob[i]:.1%} → {label[i]}"],
                marker=dict(color="rgba(255, 0, 0, 0.3)" if label[i] == "O" else "rgba(0, 0, 255, 0.3)"),
                showlegend=False
            ), row=1, col=len(target_variables) + j + 1)
        fig.update_yaxes(range=[0, 1], row=1, col=len(target_variables) + j + 1)
        
    # --- 4. t-SNE & Table コンポーネント生成 ---
    # t-SNEの計算
    existing_df = pd.DataFrame(store_data["data"])[variable_levels.keys()].dropna()
    scaler = RobustScaler().fit(existing_df)
    existing_scaled = scaler.transform(existing_df)
    recommended_scaled = scaler.transform(top_df[variable_levels.keys()])
    
    # スケーリング済みのデータを結合
    combined_scaled = np.vstack([existing_scaled, recommended_scaled])
    
    # UMAPで2次元に圧縮（UMAP前にPCAは行わない）
    n = combined_scaled.shape[0]

    if n < 4:
        embedding_combined = PCA(n_components=min(2, combined_scaled.shape[0], combined_scaled.shape[1])).fit_transform(combined_scaled)
    else:
        n_neighbors = min(15, n - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=0)
        embedding_combined = reducer.fit_transform(combined_scaled)
    
    # 既存点と推奨点に分割
    embedding_existing = embedding_combined[:len(existing_scaled)]
    embedding_recommended = embedding_combined[len(existing_scaled):]
    
    # ラベルデータの取得と整形
    label_df = pd.DataFrame(label_data["data"])
    label_df = label_df.fillna("データなし")
    hover_label = label_variable if label_variable else "ラベル"
    
    # t-SNE散布図の作成
    tsne_fig = go.Figure()

    # 既存点描画
    for i in range(len(existing_df)):
        tsne_fig.add_trace(go.Scatter(
            x=[embedding_existing[i, 0]],
            y=[embedding_existing[i, 1]],
            mode="markers",
            marker=dict(
                symbol="circle",
                color="rgba(80, 80, 80, 0.6)",  # 濃いグレー
                size=8
            ),
            name="既存点",
            legendgroup="既存点",
            text=[label_df.iloc[i][label_variable]],
            hovertemplate=(
                f"<b>{hover_label}:</b> "
                "%{text}<extra></extra>"
            ),
            showlegend=(i == 0)
        ))
    
    # 推奨点描画（順位に基づいてViridis色割り当て）
    for i in range(len(top_df)):
        tsne_fig.add_trace(go.Scatter(
            x=[embedding_recommended[i, 0]],
            y=[embedding_recommended[i, 1]],
            mode="markers",
            marker=dict(
                symbol="star",
                color=colors[i],
                size=12,
            ),
            name=f"推奨点 {i+1}",
            showlegend=True
        ))
    
    tsne_graph = dcc.Graph(
        id="tsne-graph",
        figure=tsne_fig,
        style={"width": "100%", "height": "600px"}
    )

    def format_display(val):
        try:
            abs_val = abs(float(val))
            if abs_val >= 1e3 or (abs_val > 0 and abs_val < 1e-3):
                return f"{float(val):.3e}"  # 指数表示（例：1.23e+03）
            else:
                return f"{float(val):.3f}"  # 小数点3桁
        except:
            return val  # 数値以外ならそのまま

    # 表示用データの変換
    display_df = top_df.copy()
    
    for col in top_df.columns:
        if col.endswith("_mean") or col.endswith("_std") or col.startswith("陽性確率_"):
            display_df[col] = top_df[col].apply(format_display)
    
    # 表コンポーネント
    table_component = html.Div([
        html.H5("実験推奨点"),
        html.H6(f"{num_exp} 点　/候補 {len(all_exps):,} 点"),
        dash_table.DataTable(
            data=display_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in display_df.columns],
            style_table={
                "margin-top": "10px", "display": "block", "overflowX": "auto"
            },
            style_cell={
                "textAlign": "center",
                "padding": "5px",
                "fontFamily": "Yu Gothic, Meiryo, sans-serif",
                "fontSize": "14px"
            },
            style_header={
                "fontWeight": "bold",
                "fontSize": "14px"
            },
            export_format="csv",
            export_headers="display"
        )
    ])

    return table_component, dcc.Graph(figure=fig), tsne_graph

    
def open_browser():
        webbrowser.open_new('http://127.0.0.1:8070/')
    
if __name__ == "__main__":
    freeze_support()
    Timer(1, open_browser).start()
    app.run(port=8070, debug=False)
