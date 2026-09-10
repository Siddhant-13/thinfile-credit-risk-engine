"""
Alternative-data credit risk model for thin-file borrowers.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import validation as val
from data_generator import ALTERNATIVE_FEATURES, TRADITIONAL_FEATURES
from pipeline import Config, run

# --------------------------------------------------------------------------
# Palette — tuned for a dark background. One segment scheme, every chart.
# --------------------------------------------------------------------------
TRADITIONAL = "#4d9fff"      # blue, brightened to hold up on dark
THIN_FILE = "#ff9d42"        # orange, same
SEGMENT_COLOURS = {"traditional": TRADITIONAL, "thin_file": THIN_FILE, "all": "#9aa4b0"}
TIER_SCALE = ["#3fb950", "#8fd14f", "#f2cc60", "#f0883e", "#f85149"]

PAPER = "#0e1117"            # matches Streamlit's dark background
PLOT = "#0e1117"
GRID = "#2a2f37"
TEXT = "#e6edf3"
MUTED = "#9aa4b0"
NEUTRAL = "#565f6b"          # muted bars that used to be light grey
ALERT = "#f85149"            # cut-off lines, default-rate lines
ACCENT_GREEN = "#3fb950"
ACCENT_PURPLE = "#bc8cff"

st.set_page_config(page_title="Thin-file credit risk", layout="wide")


def style(fig, height: int = 420):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor=PLOT,
        paper_bgcolor=PAPER,
        font=dict(size=13, color=TEXT),
        title=dict(font=dict(color=TEXT, size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1c2128", font=dict(color=TEXT), bordercolor=GRID),
        coloraxis_colorbar=dict(tickfont=dict(color=MUTED),
                                title=dict(font=dict(color=MUTED))),
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    # Secondary axes are declared inline on a few charts and miss the sweep above.
    for key in ("yaxis2", "xaxis2"):
        if key in fig.layout:
            fig.layout[key].update(gridcolor=GRID, linecolor=GRID,
                                   tickfont=dict(color=MUTED),
                                   title_font=dict(color=MUTED))
    # Plotly defaults bar value labels to a dark colour that vanishes here.
    fig.update_traces(selector=dict(type="bar"), textfont=dict(color=TEXT))
    # Annotation text (the IV cut-off label, the "no discrimination" line).
    for ann in fig.layout.annotations or ():
        ann.update(font=dict(color=MUTED))
    return fig


def money(x: float) -> str:
    return f"₹{x:,.0f}"


@st.cache_data(show_spinner="Generating population and fitting models…")
def load(config: Config):
    return run(config)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Parameters")

st.sidebar.caption("Population")
n_applicants = st.sidebar.slider("Applicants", 3_000, 20_000, 8_000, step=1_000)
thin_share = st.sidebar.slider("Thin-file share", 0.15, 0.65, 0.40, step=0.05)
seed = st.sidebar.number_input("Random seed", 1, 999, 7)

st.sidebar.caption("Scorecard")
iv_threshold = st.sidebar.slider("Minimum Information Value", 0.0, 0.30, 0.02, step=0.01)
max_bins = st.sidebar.slider("Maximum WOE bins per feature", 3, 10, 6)
pdo = st.sidebar.select_slider("Points to double the odds", [10, 15, 20, 25, 30, 40], value=20)

st.sidebar.caption("Gradient boosting")
tune = st.sidebar.checkbox("Search hyperparameters", value=True,
                           help="Off: use the fixed values below. On: random search on 3-fold CV.")
if tune:
    search_iterations = st.sidebar.slider("Search iterations", 5, 40, 15)
    overrides = ()
else:
    search_iterations = 5
    overrides = (
        ("n_estimators", st.sidebar.slider("Trees", 100, 800, 300, step=50)),
        ("max_depth", st.sidebar.slider("Max depth", 2, 10, 4)),
        ("learning_rate", st.sidebar.select_slider("Learning rate", [0.01, 0.02, 0.05, 0.08, 0.12, 0.2], value=0.05)),
        ("subsample", st.sidebar.slider("Subsample", 0.5, 1.0, 0.85, step=0.05)),
    )

st.sidebar.caption("Loss assumptions")
lgd = st.sidebar.slider("Loss given default", 0.10, 1.00, 0.60, step=0.05)
n_tiers = st.sidebar.slider("Risk tiers", 3, 7, 5)

config = Config(
    n_applicants=n_applicants,
    thin_file_share=thin_share,
    seed=int(seed),
    iv_threshold=iv_threshold,
    max_bins=max_bins,
    pdo=int(pdo),
    search_iterations=search_iterations,
    xgb_overrides=overrides,
    lgd=lgd,
    n_tiers=n_tiers,
)

res = load(config)
test = res["test"]
sets = res["feature_sets"]
seg_metrics = res["segment_metrics"]

st.title("Credit risk for borrowers without a credit file")
st.caption(
    "A bureau-only scorecard cannot underwrite someone who has no bureau record. "
    "This compares what traditional and alternative data can each do, "
    "for each borrower segment separately."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Population", "Scorecard", "ML models", "Segment analysis", "Risk tiers & loss"]
)

# ==========================================================================
# Tab 1 — Population
# ==========================================================================
with tab1:
    pop = res["population"]
    thin = pop[pop["segment"] == "thin_file"]
    trad = pop[pop["segment"] == "traditional"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(pop):,}")
    c2.metric("Thin-file", f"{len(thin) / len(pop):.0%}")
    c3.metric("Default rate", f"{pop['default'].mean():.1%}")
    c4.metric(
        "Thin-file default rate",
        f"{thin['default'].mean():.1%}",
        delta=f"{thin['default'].mean() - trad['default'].mean():+.1%} vs traditional",
        delta_color="inverse",
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Bureau coverage")
        coverage = []
        for col in TRADITIONAL_FEATURES + ["bnpl_ontime_rate"]:
            for name, frame in (("traditional", trad), ("thin_file", thin)):
                coverage.append(
                    {"feature": col, "segment": name, "coverage": frame[col].notna().mean()}
                )
        cov = pd.DataFrame(coverage)
        fig = px.bar(
            cov, x="coverage", y="feature", color="segment", barmode="group",
            orientation="h", color_discrete_map=SEGMENT_COLOURS,
            title="Share of applicants with the field populated",
        )
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(style(fig), width="stretch")
        st.caption(
            "This is the whole problem in one chart. Thin-file applicants have almost no "
            "bureau footprint, but their alternative-data fields are as complete as anyone's — "
            "and their small-ticket repayment history is better covered than the traditional book's."
        )

    with right:
        st.subheader("Default rate by segment")
        rate = pop.groupby("segment")["default"].agg(["mean", "size"]).reset_index()
        fig = px.bar(
            rate, x="segment", y="mean", color="segment",
            color_discrete_map=SEGMENT_COLOURS, text=rate["mean"].map("{:.1%}".format),
        )
        fig.update_yaxes(tickformat=".0%", title="default rate")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 360), width="stretch")

        st.subheader("Exposure")
        fig = px.box(pop, x="segment", y="ead", color="segment",
                     color_discrete_map=SEGMENT_COLOURS, points=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="sanctioned limit")
        st.plotly_chart(style(fig, 320), width="stretch")

    st.divider()
    st.subheader("Feature distributions")
    feature = st.selectbox("Feature", TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES, index=4)
    plot_df = pop[["segment", feature, "default"]].dropna(subset=[feature])
    d1, d2 = st.columns([2, 1])
    with d1:
        fig = px.histogram(
            plot_df, x=feature, color="segment", barmode="overlay", nbins=45,
            histnorm="probability density", color_discrete_map=SEGMENT_COLOURS,
        )
        fig.update_traces(opacity=0.65)
        st.plotly_chart(style(fig, 380), width="stretch")
    with d2:
        summary = plot_df.groupby("segment")[feature].describe()[["count", "mean", "50%", "std"]]
        st.dataframe(summary.round(2), width="stretch")
        missing = pop.groupby("segment")[feature].apply(lambda s: s.isna().mean())
        st.write("Missing rate")
        st.dataframe(missing.map("{:.1%}".format).to_frame("missing"), width="stretch")

    with st.expander("Sample rows"):
        st.dataframe(pop.head(50), width="stretch")

# ==========================================================================
# Tab 2 — Scorecard
# ==========================================================================
with tab2:
    which = st.radio("Feature set", ["augmented", "traditional"], horizontal=True,
                     format_func=lambda s: "Traditional + alternative" if s == "augmented" else "Traditional only")
    card = res["cards"][which]
    cols = sets[which]

    iv = card.binner_.iv_table()
    kept = set(card.selected_)

    c1, c2, c3 = st.columns(3)
    c1.metric("Features screened", len(iv))
    c2.metric("Features kept", len(kept))
    c3.metric("Score range", f"{int(card.score(test[cols]).min())} – {int(card.score(test[cols]).max())}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Information Value")
        iv_plot = iv.copy()
        iv_plot["kept"] = np.where(iv_plot["feature"].isin(kept), "kept", "dropped")
        fig = px.bar(
            iv_plot.sort_values("iv"), x="iv", y="feature", orientation="h", color="kept",
            color_discrete_map={"kept": TRADITIONAL, "dropped": NEUTRAL},
        )
        fig.add_vline(x=iv_threshold, line_dash="dash", line_color=ALERT,
                      annotation_text=f"cut-off {iv_threshold:.2f}")
        st.plotly_chart(style(fig, 460), width="stretch")
        st.caption(
            "IVs on synthetic data run high — on a real book anything above 0.5 usually "
            "means a leaked outcome variable, not a great feature."
        )
    with right:
        st.subheader("How much each feature can move a score")
        swing = card.points_range()
        fig = px.bar(swing.sort_values("swing"), x="swing", y="feature", orientation="h",
                     color_discrete_sequence=[THIN_FILE])
        fig.update_xaxes(title="points between the best and worst bucket")
        st.plotly_chart(style(fig, 460), width="stretch")

    st.divider()
    st.subheader("WOE bins")
    feat = st.selectbox("Feature", card.selected_, key="woe_feature")
    table = card.binner_.tables_[feat]
    b1, b2 = st.columns([1, 1])
    with b1:
        fig = go.Figure()
        fig.add_bar(x=table["bucket"].astype(str), y=table["woe"],
                    marker_color=np.where(table["woe"] >= 0, TRADITIONAL, THIN_FILE), name="WOE")
        fig.update_layout(title=f"WOE by bucket — {feat}")
        fig.update_yaxes(title="WOE (higher = safer)")
        st.plotly_chart(style(fig, 340), width="stretch")
    with b2:
        fig = go.Figure()
        fig.add_bar(x=table["bucket"].astype(str), y=table["population_pct"],
                    marker_color=NEUTRAL, name="population share")
        fig.add_scatter(x=table["bucket"].astype(str), y=table["bad_rate"], yaxis="y2",
                        mode="lines+markers", line=dict(color=ALERT), name="default rate")
        fig.update_layout(title=f"Population and default rate — {feat}",
                          yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
        st.plotly_chart(style(fig, 340), width="stretch")

    st.dataframe(
        table[["bucket", "count", "population_pct", "bad_rate", "woe", "iv"]].round(4),
        width="stretch",
    )

    st.divider()
    st.subheader("Published scorecard")
    st.dataframe(card.scorecard_table().round(3), width="stretch", height=340)

    st.divider()
    st.subheader("Score an applicant")
    st.caption("Leave a field at its 'not available' setting to score them as a thin-file applicant.")

    inputs = {}
    grid = st.columns(3)
    for i, col in enumerate(card.selected_):
        if col.endswith("_missing"):
            continue
        with grid[i % 3]:
            available = st.checkbox(f"{col} available", value=True, key=f"has_{col}")
            series = res["population"][col].dropna()
            if available:
                inputs[col] = st.number_input(
                    col,
                    float(series.min()), float(series.max()), float(series.median()),
                    key=f"val_{col}",
                )
            else:
                inputs[col] = np.nan

    for col in card.selected_:
        if col.endswith("_missing"):
            base = col.replace("_missing", "")
            inputs[col] = int(pd.isna(inputs.get(base, np.nan)))

    breakdown, total, pd_est = card.explain(inputs)
    m1, m2, m3 = st.columns(3)
    m1.metric("Score", int(total))
    m2.metric("Probability of default", f"{pd_est:.2%}")
    m3.metric("Expected loss", money(pd_est * lgd * float(res["population"]["ead"].median())),
              help="At the median exposure and the sidebar LGD.")

    fig = px.bar(breakdown.sort_values("points"), x="points", y="feature", orientation="h",
                 color="points", color_continuous_scale=["#f85149", "#f2cc60", "#3fb950"],
                 hover_data=["value", "bucket", "woe"])
    fig.update_layout(coloraxis_showscale=False, title="Where the points came from")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.dataframe(breakdown, width="stretch")

# ==========================================================================
# Tab 3 — ML models
# ==========================================================================
with tab3:
    overall = res["overall_metrics"]
    st.subheader("Performance on the held-out set")
    st.dataframe(
        overall[["model", "n", "auc", "gini", "ks", "precision_default", "recall_default",
                 "f1_default", "accuracy"]].round(3),
        width="stretch",
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("ROC curves")
        fig = go.Figure()
        palette = {
            "XGBoost (augmented)": ACCENT_GREEN,
            "LightGBM (augmented)": ACCENT_PURPLE,
            "Scorecard (augmented)": TRADITIONAL,
            "Scorecard (traditional)": THIN_FILE,
            "XGBoost (traditional)": NEUTRAL,
        }
        for name, points in res["roc"].items():
            auc_value = float(overall.loc[overall["model"] == name, "auc"].iloc[0])
            fig.add_scatter(x=points["fpr"], y=points["tpr"], mode="lines",
                            name=f"{name} — {auc_value:.3f}",
                            line=dict(color=palette.get(name), width=2.5))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                        line=dict(dash="dot", color=MUTED))
        fig.update_xaxes(title="false positive rate")
        fig.update_yaxes(title="true positive rate")
        st.plotly_chart(style(fig, 480), width="stretch")
    with right:
        st.subheader("Tuning")
        for key, label in (("xgboost", "XGBoost"), ("lightgbm", "LightGBM")):
            model = res["models"][key]
            st.write(f"**{label}** — CV AUC {model.cv_auc:.3f}" if model.cv_auc == model.cv_auc
                     else f"**{label}** — fixed parameters")
            st.json(model.params, expanded=False)

    st.divider()
    st.subheader("SHAP feature importance — XGBoost")
    imp = res["shap"]["importance"]
    s1, s2 = st.columns([1, 1])
    with s1:
        fig = px.bar(imp.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                     orientation="h", color_discrete_sequence=[ACCENT_GREEN])
        fig.update_xaxes(title="mean |SHAP| (log-odds)")
        st.plotly_chart(style(fig, 480), width="stretch")
    with s2:
        by_seg = res["shap"]["by_segment"]
        top = imp.head(10)["feature"].tolist()
        fig = px.bar(
            by_seg[by_seg["feature"].isin(top)], x="share_of_attribution", y="feature",
            color="segment", barmode="group", orientation="h",
            color_discrete_map=SEGMENT_COLOURS,
            title="Share of attribution within each segment",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 480), width="stretch")
        st.caption(
            "The same model leans on different evidence for each segment: bureau fields for "
            "traditional borrowers, behavioural fields for thin-file ones. Nothing told it to — "
            "it is reading the missingness."
        )

    st.divider()
    st.subheader("Explainability against performance")
    st.dataframe(res["tradeoff"].round(4), width="stretch")

    t1, t2 = st.columns([1, 1])
    with t1:
        trade = res["tradeoff"]
        fig = px.bar(trade, x="model", y="auc", color="auditable_by_hand",
                     color_discrete_map={"yes": TRADITIONAL, "no": NEUTRAL},
                     text=trade["auc"].map("{:.3f}".format))
        fig.update_yaxes(range=[0.5, 1.0], title="AUC")
        fig.update_layout(title="What the extra complexity buys")
        st.plotly_chart(style(fig, 380), width="stretch")
    with t2:
        st.dataframe(res["rank_agreement"].round(3), width="stretch", height=380)
        st.caption(
            "Where the two models rank features differently, one of them is wrong about the "
            "business — that gap is the first thing to take to a model-risk review."
        )

    uplift = float(res["tradeoff"]["auc_uplift_vs_scorecard"].max())
    st.info(
        f"Gradient boosting adds {uplift:+.3f} AUC over the scorecard on the same features. "
        "The scorecard is a page of point buckets a validator can check by hand; the boosted "
        "model is several hundred trees with SHAP attributions bolted on afterwards. Whether "
        f"{uplift:.3f} AUC is worth that depends on whether the lender has to explain declines."
    )

    st.divider()
    st.subheader("Stability (PSI)")
    p1, p2 = st.columns([1, 2])
    with p1:
        value = res["psi"]["score_psi"]
        st.metric("Score PSI, vintage 1 vs 2", f"{value:.4f}", val.psi_verdict(value))
        st.caption("Below 0.10 stable · 0.10–0.25 monitor · above 0.25 investigate.")
        st.dataframe(res["psi"]["features"].round(4), width="stretch", height=300)
    with p2:
        detail = res["psi"]["detail"]
        fig = go.Figure()
        fig.add_bar(x=detail["bucket"], y=detail["expected_pct"], name="vintage 1",
                    marker_color=TRADITIONAL)
        fig.add_bar(x=detail["bucket"], y=detail["actual_pct"], name="vintage 2",
                    marker_color=THIN_FILE)
        fig.update_layout(barmode="group", title="Score distribution by vintage")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 420), width="stretch")

# ==========================================================================
# Tab 4 — Segment analysis (the point of the project)
# ==========================================================================
with tab4:
    st.subheader("The same models, judged separately on each segment")

    focus = ["Scorecard (traditional)", "Scorecard (augmented)",
             "XGBoost (traditional)", "XGBoost (augmented)"]
    view = seg_metrics[seg_metrics["model"].isin(focus)].copy()

    thin_trad = view.query("model == 'Scorecard (traditional)' and segment == 'thin_file'")["auc"].iloc[0]
    thin_aug = view.query("model == 'Scorecard (augmented)' and segment == 'thin_file'")["auc"].iloc[0]
    trad_trad = view.query("model == 'Scorecard (traditional)' and segment == 'traditional'")["auc"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Bureau-only AUC on thin-file", f"{thin_trad:.3f}",
              help="0.50 is a coin toss.")
    c2.metric("With alternative data", f"{thin_aug:.3f}", delta=f"{thin_aug - thin_trad:+.3f}")
    c3.metric("Bureau-only AUC on traditional borrowers", f"{trad_trad:.3f}",
              help="The same feature set works fine here — the failure is segment-specific.")

    fig = px.bar(
        view, x="model", y="auc", color="segment", barmode="group",
        color_discrete_map=SEGMENT_COLOURS, text=view["auc"].map("{:.3f}".format),
        category_orders={"model": focus, "segment": ["all", "traditional", "thin_file"]},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color=ALERT,
                  annotation_text="no discrimination")
    fig.update_yaxes(range=[0.4, 1.0], title="AUC")
    fig.update_layout(title="AUC by model and borrower segment")
    st.plotly_chart(style(fig, 480), width="stretch")

    st.warning(
        f"A bureau-only scorecard scores {trad_trad:.3f} AUC on borrowers who have a bureau file "
        f"and {thin_trad:.3f} on borrowers who do not — statistically indistinguishable from "
        "guessing. Adding alternative data recovers that segment to "
        f"{thin_aug:.3f}. The bureau model is not a weak model; it is a model applied to people "
        "it has no information about."
    )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Catching defaults in the thin-file book")
        thin_view = seg_metrics[
            (seg_metrics["segment"] == "thin_file") & (seg_metrics["model"].isin(focus))
        ]
        melted = thin_view.melt(
            id_vars="model", value_vars=["recall_default", "precision_default", "ks"],
            var_name="metric", value_name="value",
        )
        fig = px.bar(melted, x="metric", y="value", color="model", barmode="group",
                     category_orders={"model": focus})
        st.plotly_chart(style(fig, 400), width="stretch")
    with right:
        st.subheader("ROC on the thin-file segment only")
        y_thin = test["default"][test["segment"] == "thin_file"]
        mask = (test["segment"] == "thin_file").values
        fig = go.Figure()
        for name in focus:
            points = val.roc_points(y_thin, np.asarray(res["scores"][name])[mask])
            fig.add_scatter(x=points["fpr"], y=points["tpr"], mode="lines", name=name)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                        line=dict(dash="dot", color=MUTED))
        st.plotly_chart(style(fig, 400), width="stretch")

    st.divider()
    st.subheader("Full segment table")
    st.dataframe(
        seg_metrics[["model", "segment", "n", "defaults", "default_rate", "auc", "gini", "ks",
                     "precision_default", "recall_default", "f1_default"]].round(3),
        width="stretch", height=420,
    )

    st.subheader("Approval consequences")
    st.caption(
        "If the bureau-only model is used as the gate, thin-file applicants are ranked at random, "
        "so they get declined roughly in proportion to how conservative the cut-off is — "
        "regardless of whether they would have repaid."
    )
    cut = st.slider("Approval rate", 0.30, 1.00, 0.70, step=0.05)
    rows = []
    for name in focus:
        score = np.asarray(res["scores"][name])
        threshold = np.quantile(score, cut)
        approved = test[score <= threshold]
        rows.append(
            {
                "model": name,
                "approved": len(approved),
                "thin_file_share_of_approvals": (approved["segment"] == "thin_file").mean(),
                "default_rate_of_approved_book": approved["default"].mean(),
                "thin_file_default_rate_if_approved": approved.loc[
                    approved["segment"] == "thin_file", "default"].mean(),
            }
        )
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch")

# ==========================================================================
# Tab 5 — Risk tiers and expected loss
# ==========================================================================
with tab5:
    tiers = res["tiers"]
    portfolio = res["portfolio"]
    calib = res["calibration"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposure at default", money(portfolio["total_exposure"]))
    c2.metric("Expected loss", money(portfolio["total_expected_loss"]))
    c3.metric("EL rate on exposure", f"{portfolio['expected_loss_rate']:.2%}")
    c4.metric("Realised loss on the test book", money(portfolio.get("realised_loss", 0)))

    st.caption(
        f"Model: {res['best_model']}. LGD fixed at {lgd:.0%} — an assumption, not an estimate; "
        "nothing here models recoveries. EAD is the sanctioned limit."
    )

    ratio = calib["calibrated"]["ratio"]
    if abs(ratio - 1) > 0.15:
        st.warning(
            f"Average predicted PD is {calib['calibrated']['mean_predicted_pd']:.2%} against an "
            f"observed default rate of {calib['calibrated']['observed_default_rate']:.2%} "
            f"(ratio {ratio:.2f}). Class weighting was reversed out before these numbers were "
            f"computed — without that step the ratio is {calib['raw']['ratio']:.2f} and portfolio "
            "EL would be overstated several times over. The residual gap would need a proper "
            "calibration layer before any of this reached a pricing committee."
        )

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Expected loss by tier")
        fig = go.Figure()
        fig.add_bar(x=tiers["tier"], y=tiers["expected_loss"],
                    marker_color=TIER_SCALE[: len(tiers)], name="expected loss")
        fig.add_scatter(x=tiers["tier"], y=tiers["avg_pd"], yaxis="y2", mode="lines+markers",
                        name="average PD", line=dict(color=TEXT))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".1%",
                                      title="average PD"))
        fig.update_yaxes(title="expected loss")
        st.plotly_chart(style(fig, 420), width="stretch")
    with right:
        st.subheader("Predicted against realised")
        fig = go.Figure()
        fig.add_bar(x=tiers["tier"], y=tiers["avg_pd"], name="predicted PD", marker_color=TRADITIONAL)
        fig.add_bar(x=tiers["tier"], y=tiers["actual_default_rate"], name="realised default rate",
                    marker_color=THIN_FILE)
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 420), width="stretch")

    st.dataframe(
        tiers[["tier", "borrowers", "avg_pd", "min_pd", "max_pd", "lgd", "avg_ead", "total_ead",
               "expected_loss", "el_rate_on_exposure", "share_of_expected_loss",
               "actual_default_rate"]].round(4),
        width="stretch",
    )

    st.divider()
    s1, s2 = st.columns([1, 1])
    with s1:
        st.subheader("Expected loss by segment")
        el_seg = res["el_segment"]
        fig = px.bar(el_seg, x="segment", y="expected_loss", color="segment",
                     color_discrete_map=SEGMENT_COLOURS,
                     text=el_seg["expected_loss"].map(lambda v: money(v)))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 360), width="stretch")
        st.dataframe(el_seg.round(4), width="stretch")
        st.caption(
            "Thin-file borrowers carry a higher PD on a much smaller ticket. Whether they are a "
            "worse book depends on pricing, not on the default rate alone."
        )
    with s2:
        st.subheader("Where to set the cut-off")
        cutoffs = res["cutoffs"]
        fig = go.Figure()
        fig.add_scatter(x=cutoffs["approval_rate"], y=cutoffs["el_rate"], mode="lines+markers",
                        name="EL rate on approved exposure", line=dict(color=ALERT))
        fig.add_scatter(x=cutoffs["approval_rate"], y=cutoffs["thin_file_share_of_approvals"],
                        mode="lines+markers", name="thin-file share of approvals",
                        line=dict(color=THIN_FILE))
        fig.update_xaxes(title="approval rate", tickformat=".0%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(style(fig, 360), width="stretch")
        st.dataframe(cutoffs.round(4), width="stretch", height=260)