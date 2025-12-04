import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import os
# from pyngrok import ngrok
# ngrok.set_auth_token("36Km6IAoX1cnWOIwnDWnvq7HUWS_3ifpQBuUCk9P3tgAZsMhY")
# -------------------------------
# 1. Load dữ liệu Excel
# -------------------------------
folder = r"C:\Users\TS-1106\OneDrive\KATE_TL0801\MECE\Factor_Impact"
df = pd.read_excel(f"{folder}/Instant_raw_SGN.xlsx")

# =========================================
# 2. Dash App
# =========================================
app = Dash(__name__)
app.title = "Acceptance Rate Dashboard"

# =========================================
# 3. Layout
# =========================================
app.layout = html.Div([
    html.H1("Dashboard: AR, Impact AR, và rq_stp theo Surcharge & Order Hour (%)"),

    html.Div([
        html.Label("Service Type"),
        dcc.Dropdown(
            options=[{"label": t, "value": t} for t in df["ServiceType"].dropna().unique()],
            id="service-type",
            placeholder="Chọn Service Type"
        ),

        html.Label("Service ID"),
        dcc.Dropdown(
            options=[{"label": s, "value": s} for s in df["service_id"].dropna().unique()],
            id="service-id",
            placeholder="Chọn Service ID"
        ),

        html.Label("District"),
        dcc.Dropdown(
            options=[{"label": d, "value": d} for d in df["District"].dropna().unique()],
            id="district",
            placeholder="Chọn District"
        ),

        html.Label("Ngày (Period)"),
        dcc.DatePickerSingle(id="period", date=None)
    ], style={"columnCount": 2, "padding": 10}),

    html.Hr(),

    html.H2("Heatmap Acceptance Rate (%)"),
    dcc.Graph(id="heatmap-ar"),
    html.Button("Download AR Table", id="btn-download-ar"),
    dcc.Download(id="download-ar-table"),
    html.Div(id="ar-table"),

    html.Hr(),

    html.H2("Heatmap Impact AR (%)"),
    dcc.Graph(id="heatmap-impact-ar"),
    html.Button("Download Impact AR Table", id="btn-download-impact"),
    dcc.Download(id="download-impact-ar-table"),
    html.Div(id="impact-ar-table"),

    html.Hr(),

    html.H2("Heatmap Số lượng đơn hàng (rq_stp)"),
    dcc.Graph(id="heatmap-rq"),
    html.Button("Download rq_stp Table", id="btn-download-rq"),
    dcc.Download(id="download-rq-table"),
    html.Div(id="rq-table"),

    html.Hr(),


    # ===========================
    # --- NEW CHART 1: CORRELATION HEATMAP ---
    # ===========================
    html.H2("Heatmap tương quan Surcharge ↔ AR theo từng Order Hour"),
    dcc.Graph(id="corr-heatmap"),

    html.Hr(),

    # ===========================
    # --- NEW CHART 2: HOURLY SCATTER ---
    # ===========================
    html.H2("Scatter theo từng giờ (click từ heatmap)"),
    dcc.Graph(id="scatter-hourly"),
])



# =========================================
# 4. Function pivot
# =========================================
def prepare_pivot(dff, value_col):
    dff = dff.dropna(subset=["surcharge", "rq_stp", "accept_stp"])
    if dff.empty:
        dff = df.dropna(subset=["surcharge", "rq_stp", "accept_stp"]).copy()

    min_s = dff["surcharge"].min()
    max_s = dff["surcharge"].max()
    bin_step = 0.05
    bins = np.arange(min_s, max_s + bin_step, bin_step)
    labels = [f"{b:.2f}-{b+bin_step:.2f}" for b in bins[:-1]]
    dff["surcharge_bin"] = pd.cut(dff["surcharge"], bins=bins, labels=labels, include_lowest=True)

    dff["order_hour"] = dff["order_hour"].astype(int)
    all_hours = np.arange(24)
    full_index = pd.MultiIndex.from_product([all_hours, labels], names=["order_hour", "surcharge_bin"])

    pivot = dff.groupby(["order_hour", "surcharge_bin"]).agg(
        accept_stp_sum=("accept_stp", "sum"),
        rq_stp_sum=("rq_stp", "sum")
    ).reindex(full_index).reset_index()

    if value_col == "AcceptRate":
        pivot["AcceptRate"] = np.where(
            pivot["rq_stp_sum"] > 0,
            pivot["accept_stp_sum"] / pivot["rq_stp_sum"] * 100,
            np.nan
        )
    elif value_col == "ImpactAR":
        total_rq = pivot["rq_stp_sum"].sum()
        pivot["ImpactAR"] = np.where(
            (pivot["rq_stp_sum"] > 0) & (total_rq > 0),
            pivot["accept_stp_sum"] / total_rq * 100,
            np.nan
        )
    elif value_col == "rq_stp":
        pivot["rq_stp_val"] = pivot["rq_stp_sum"]

    return pivot, labels, all_hours

# =========================================
# 5. Callback AR
# =========================================
@app.callback(
    [Output("heatmap-ar", "figure"),
     Output("ar-table", "children")],
    [Input("service-type", "value"),
     Input("service-id", "value"),
     Input("district", "value"),
     Input("period", "date")]
)
def update_ar(service_type, service_id, district, period):
    dff = df.copy()
    if service_type: dff = dff[dff["ServiceType"] == service_type]
    if service_id: dff = dff[dff["service_id"] == service_id]
    if district: dff = dff[dff["District"] == district]
    if period: dff = dff[dff["period"] == pd.to_datetime(period)]

    pivot, labels, all_hours = prepare_pivot(dff, "AcceptRate")
    heat_matrix = pivot.pivot_table(index="surcharge_bin", columns="order_hour",
                                    values="AcceptRate").reindex(columns=all_hours)

    heat_matrix_str = heat_matrix.round(1).astype(object)
    heat_matrix_str = heat_matrix_str.applymap(lambda x: "" if pd.isna(x) else f"{x}%")

    fig = go.Figure(data=go.Heatmap(
        z=np.nan_to_num(heat_matrix.values, nan=0),
        x=heat_matrix.columns,
        y=heat_matrix.index,
        colorscale='YlGnBu',
        text=heat_matrix_str.values,
        texttemplate="%{text}"
    ))

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(24)),
        ticktext=[str(i) for i in range(24)],
        automargin=True
    )

    fig.update_layout(
        width=1500,  # <-- quan trọng để hiển thị đủ nhãn
        xaxis_title="Order Hour",
        yaxis_title="Surcharge Bin",
        yaxis_autorange='reversed',
        margin=dict(l=50, r=20, t=40, b=80)
    )

    table_html = html.Table([
        html.Thead(html.Tr([html.Th("Surcharge Bin")] +
                           [html.Th(str(h)) for h in heat_matrix.columns])),
        html.Tbody([
            html.Tr([html.Td(idx)] + [
                html.Td("" if pd.isna(heat_matrix.loc[idx, col]) else f"{round(heat_matrix.loc[idx, col],1)}%")
                for col in heat_matrix.columns
            ]) for idx in heat_matrix.index
        ])
    ])

    return fig, table_html


# =========================================
# 6. Callback Impact AR
# =========================================
@app.callback(
    [Output("heatmap-impact-ar", "figure"),
     Output("impact-ar-table", "children")],
    [Input("service-type", "value"),
     Input("service-id", "value"),
     Input("district", "value"),
     Input("period", "date")]
)
def update_impact(service_type, service_id, district, period):
    dff = df.copy()
    if service_type: dff = dff[dff["ServiceType"] == service_type]
    if service_id: dff = dff[dff["service_id"] == service_id]
    if district: dff = dff[dff["District"] == district]
    if period: dff = dff[dff["period"] == pd.to_datetime(period)]

    pivot, labels, all_hours = prepare_pivot(dff, "ImpactAR")
    heat_matrix = pivot.pivot_table(index="surcharge_bin", columns="order_hour",
                                    values="ImpactAR").reindex(columns=all_hours)

    heat_matrix_str = heat_matrix.round(1).astype(object)
    heat_matrix_str = heat_matrix_str.applymap(lambda x: "" if pd.isna(x) else f"{x}%")

    fig = go.Figure(data=go.Heatmap(
        z=np.nan_to_num(heat_matrix.values, nan=0),
        x=heat_matrix.columns,
        y=heat_matrix.index,
        colorscale='YlOrRd',
        text=heat_matrix_str.values,
        texttemplate="%{text}"
    ))

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(24)),
        ticktext=[str(i) for i in range(24)],
        automargin=True
    )

    fig.update_layout(
        width=1500,
        xaxis_title="Order Hour",
        yaxis_title="Surcharge Bin",
        yaxis_autorange='reversed',
        margin=dict(l=50, r=20, t=40, b=80)
    )

    table_html = html.Table([
        html.Thead(html.Tr([html.Th("Surcharge Bin")] +
                           [html.Th(str(h)) for h in heat_matrix.columns])),
        html.Tbody([
            html.Tr([html.Td(idx)] + [
                html.Td("" if pd.isna(heat_matrix.loc[idx, col]) else f"{round(heat_matrix.loc[idx, col],1)}%")
                for col in heat_matrix.columns
            ]) for idx in heat_matrix.index
        ])
    ])

    return fig, table_html


# =========================================
# 7. Callback rq_stp
# =========================================
@app.callback(
    [Output("heatmap-rq", "figure"),
     Output("rq-table", "children")],
    [Input("service-type", "value"),
     Input("service-id", "value"),
     Input("district", "value"),
     Input("period", "date")]
)
def update_rq(service_type, service_id, district, period):
    dff = df.copy()
    if service_type: dff = dff[dff["ServiceType"] == service_type]
    if service_id: dff = dff[dff["service_id"] == service_id]
    if district: dff = dff[dff["District"] == district]
    if period: dff = dff[dff["period"] == pd.to_datetime(period)]

    pivot, labels, all_hours = prepare_pivot(dff, "rq_stp")
    heat_matrix = pivot.pivot_table(index="surcharge_bin", columns="order_hour",
                                    values="rq_stp_val").reindex(columns=all_hours)

    heat_matrix_str = heat_matrix.fillna(0).astype(int).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=np.nan_to_num(heat_matrix.values, nan=0),
        x=heat_matrix.columns,
        y=heat_matrix.index,
        colorscale='Blues',
        text=heat_matrix_str.values,
        texttemplate="%{text}"
    ))

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(24)),
        ticktext=[str(i) for i in range(24)],
        automargin=True
    )

    fig.update_layout(
        width=1500,
        xaxis_title="Order Hour",
        yaxis_title="Surcharge Bin",
        yaxis_autorange='reversed',
        margin=dict(l=50, r=20, t=40, b=80)
    )

    table_html = html.Table([
        html.Thead(html.Tr([html.Th("Surcharge Bin")] +
                           [html.Th(str(h)) for h in heat_matrix.columns])),
        html.Tbody([
            html.Tr([html.Td(idx)] + [
                html.Td("" if pd.isna(heat_matrix.loc[idx, col]) else int(heat_matrix.loc[idx, col]))
                for col in heat_matrix.columns
            ]) for idx in heat_matrix.index
        ])
    ])

    return fig, table_html




# ============================================================
# ===============  NEW CALLBACK 1: CORRELATION HEATMAP  ======
# ============================================================
@app.callback(
    Output("corr-heatmap", "figure"),
    [
        Input("service-type", "value"),
        Input("service-id", "value"),
        Input("district", "value"),
        Input("period", "date")
    ]
)
def update_correlation_heatmap(service_type, service_id, district, period):
    dff = df.copy()

    if service_type:
        dff = dff[dff["ServiceType"] == service_type]
    if service_id:
        dff = dff[dff["service_id"] == service_id]
    if district:
        dff = dff[dff["District"] == district]
    if period:
        dff = dff[dff["period"] == pd.to_datetime(period)]

    dff = dff.dropna(subset=["surcharge", "rq_stp", "accept_stp", "order_hour"])
    dff["AR"] = np.where(dff["rq_stp"] > 0,
                         dff["accept_stp"] / dff["rq_stp"] * 100,
                         np.nan)

    corrs = []
    hours = list(range(24))

    for h in hours:
        temp = dff[dff["order_hour"] == h]
        if temp["surcharge"].nunique() > 1 and temp["AR"].notna().sum() > 2:
            corrs.append(temp["surcharge"].corr(temp["AR"]))
        else:
            corrs.append(np.nan)

    fig = go.Figure(data=go.Heatmap(
        z=[corrs],
        x=hours,
        y=["Correlation"],
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Corr")
    ))

    fig.update_layout(
        title="Tương quan giữa Surcharge và AR theo từng Order Hour",
        xaxis_title="Order Hour",
        height=300
    )
    return fig


# ============================================================
# ===============  NEW CALLBACK 2: HOURLY SCATTER  ===========
# ============================================================
@app.callback(
    Output("scatter-hourly", "figure"),
    Input("corr-heatmap", "clickData"),
    [
        Input("service-type", "value"),
        Input("service-id", "value"),
        Input("district", "value"),
        Input("period", "date")
    ]
)
def update_hourly_scatter(clickData, service_type, service_id, district, period):

    if clickData is None:
        return go.Figure()

    hour_clicked = int(clickData["points"][0]["x"])

    dff = df.copy()

    if service_type:
        dff = dff[dff["ServiceType"] == service_type]
    if service_id:
        dff = dff[dff["service_id"] == service_id]
    if district:
        dff = dff[dff["District"] == district]
    if period:
        dff = dff[dff["period"] == pd.to_datetime(period)]

    dff = dff[dff["order_hour"] == hour_clicked]
    dff = dff.dropna(subset=["surcharge", "rq_stp", "accept_stp"])

    dff["AR"] = np.where(dff["rq_stp"] > 0,
                         dff["accept_stp"] / dff["rq_stp"] * 100,
                         np.nan)

    fig = go.Figure()

    # Scatter
    fig.add_trace(go.Scatter(
        x=dff["surcharge"],
        y=dff["AR"],
        mode="markers",
        marker=dict(size=8, color="blue"),
        name="Data"
    ))

    # Trendline
    clean = dff.dropna(subset=["surcharge", "AR"])
    if len(clean) > 2:
        coef = np.polyfit(clean["surcharge"], clean["AR"], 1)
        poly = np.poly1d(coef)

        fig.add_trace(go.Scatter(
            x=clean["surcharge"],
            y=poly(clean["surcharge"]),
            mode="lines",
            line=dict(color="red"),
            name=f"Trendline (slope={coef[0]:.2f})"
        ))

    fig.update_layout(
        title=f"Surcharge → AR tại giờ {hour_clicked}",
        xaxis_title="Surcharge",
        yaxis_title="AR (%)",
        height=500
    )

    return fig



# =========================================
# 9. Run App
# =========================================
# if __name__ == "__main__":
#     app.run(debug=True)
# public_url = ngrok.connect(8050)
# print("Public URL:", public_url)
# app.run(host="0.0.0.0", port=8050)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Render sẽ cung cấp PORT
    app.run(host="0.0.0.0", port=port, debug=True)