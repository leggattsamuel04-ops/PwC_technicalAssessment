from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import gaussian_kde


BASE_DIR = Path(__file__).parent
RAW_DATA_PATH = BASE_DIR / "steam_reviews 1.csv"
GAME_FEATURES_PATH = BASE_DIR / "game_level_features.csv"
REVIEW_SENTIMENT_PATH = BASE_DIR / "review_level_sentiment.csv"
GAME_REVIEWS_PATH = BASE_DIR / "game_reviews_table.json"


st.set_page_config(
    page_title="GameVault Acquisition Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    .small-note {
        color: #52525b;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .decision-box {
        border: 1px solid #d4d4d8;
        border-radius: 8px;
        padding: 14px 16px;
        background: #fafafa;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(RAW_DATA_PATH)
    games = pd.read_csv(GAME_FEATURES_PATH)
    reviews = pd.read_csv(REVIEW_SENTIMENT_PATH)

    raw["date_posted"] = pd.to_datetime(raw["date_posted"], errors="coerce")
    reviews["date_posted"] = pd.to_datetime(reviews["date_posted"], errors="coerce")
    games["first_review"] = pd.to_datetime(games["first_review"], errors="coerce")
    games["last_review"] = pd.to_datetime(games["last_review"], errors="coerce")

    with open(GAME_REVIEWS_PATH) as f:
        game_reviews = json.load(f)

    return raw, games, reviews, game_reviews


def minmax(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if min_value == max_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def build_kde_frame(data: pd.DataFrame, columns: list[str], value_column: str = "value") -> pd.DataFrame:
    rows = []
    for column in columns:
        values = data[column].dropna().astype(float)
        if len(values) < 2:
            continue

        x_min = values.min()
        x_max = values.max()
        if x_min == x_max:
            x_grid = np.array([x_min])
            density = np.array([1.0])
        else:
            padding = (x_max - x_min) * 0.08
            x_grid = np.linspace(x_min - padding, x_max + padding, 200)
            density = gaussian_kde(values)(x_grid)

        rows.append(
            pd.DataFrame(
                {
                    "feature": column,
                    value_column: x_grid,
                    "density": density,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["feature", value_column, "density"])
    return pd.concat(rows, ignore_index=True)


def ranking_component_long(games: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    component_cols = [
        "approval_score",
        "engagement_score",
        "evidence_score",
        "longevity_score",
        "review_quality_score",
    ]
    top_titles = games.sort_values("opportunity_rank").head(top_n)["title"].tolist()
    long_df = games.melt(
        id_vars=["title", "opportunity_rank", "opportunity_tier"],
        value_vars=component_cols,
        var_name="score_component",
        value_name="component_value",
    )
    return long_df[long_df["title"].isin(top_titles)]


def risk_text(row: pd.Series) -> str:
    flags = []
    if row.get("risk_flag_low_review_count", False):
        flags.append("low review count")
    if row.get("risk_flag_short_review_span", False):
        flags.append("short review span")
    if row.get("risk_flag_mixed_approval", False):
        flags.append("mixed approval")
    return ", ".join(flags) if flags else "no major flags"


raw_df, game_df, review_df, game_reviews_table = load_data()
game_df = game_df.sort_values("opportunity_rank")


st.sidebar.title("GameVault")
top_n = st.sidebar.slider("Top candidates shown", min_value=5, max_value=20, value=10)


st.title("GameVault Acquisition Intelligence")
st.caption(
    "Centralised application for ranking games by acquisition potential"
)


raw_tab, features_tab, eda_tab, overview_tab, sentiment_tab, scoring_tab, business_tab = st.tabs(
    [
        "Raw Reviews",
        "Game Features",
        "EDA & Scaling",
        "Executive View",
        "Sentiment",
        "Scoring",
        "Business Value",
    ]
)


with overview_tab:
    st.subheader("Prioritised Acquisition Shortlist")

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Reviews analysed", f"{len(raw_df):,}")
    kpi_cols[1].metric("Unique games", f"{game_df['title'].nunique():,}")
    kpi_cols[2].metric("Top score", f"{game_df['final_score'].max():.3f}")
    kpi_cols[3].metric(
        "Median recommendation",
        format_percent(game_df["recommended_proportion"].median()),
    )
    kpi_cols[4].metric("Priority targets", f"{(game_df['opportunity_tier'] == 'Priority Target').sum():,}")

    display_cols = [
        "opportunity_rank",
        "title",
        "opportunity_tier",
        "final_score",
        "approval_score",
        "engagement_score",
        "evidence_score",
        "longevity_score",
        "review_quality_score",
        "risk_flag_count",
    ]
    st.dataframe(
        game_df[display_cols].head(top_n).style.format(
            {
                "final_score": "{:.3f}",
                "approval_score": "{:.3f}",
                "engagement_score": "{:.3f}",
                "evidence_score": "{:.3f}",
                "longevity_score": "{:.3f}",
                "review_quality_score": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        components = ranking_component_long(game_df, top_n=top_n)
        top_titles = game_df.head(top_n)["title"].tolist()
        fig = px.bar(
            components,
            x="component_value",
            y="title",
            color="score_component",
            orientation="h",
            title="Score Component Breakdown",
            category_orders={"title": list(reversed(top_titles))},
            labels={"component_value": "Scaled component score", "title": ""},
        )
        fig.update_layout(height=520, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.scatter(
            game_df,
            x="engagement_score",
            y="approval_score",
            size="review_count",
            color="opportunity_tier",
            hover_name="title",
            hover_data=[
                "opportunity_rank",
                "final_score",
                "evidence_score",
                "longevity_score",
                "risk_flag_count",
            ],
            title="Candidate Map: Approval vs Engagement",
            labels={
                "engagement_score": "Engagement score",
                "approval_score": "Approval score",
            },
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="small-note">Bubble size represents review count. Colour represents '
            "opportunity tier. Games further up/right combine stronger approval and engagement.</div>",
            unsafe_allow_html=True,
        )


with raw_tab:
    st.subheader("Original Review-Level Dataset")
    st.markdown(
        '<div class="small-note">Each row is one Steam review. The acquisition decision, however, '
        "is game-level, so this raw table is transformed later into one row per game.</div>",
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(raw_df):,}")
    metric_cols[1].metric("Columns", f"{raw_df.shape[1]:,}")
    metric_cols[2].metric("Missing review text", f"{raw_df['review'].isna().sum():,}")
    metric_cols[3].metric("Duplicate rows", f"{raw_df.duplicated().sum():,}")

    st.dataframe(raw_df.head(500), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        rec_counts = raw_df["recommendation"].value_counts(normalize=True).reset_index()
        rec_counts.columns = ["recommendation", "share"]
        fig = px.bar(
            rec_counts,
            x="recommendation",
            y="share",
            title="Recommendation Split",
            text_auto=".1%",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        reviews_by_game = raw_df["title"].value_counts().reset_index()
        reviews_by_game.columns = ["title", "review_count"]
        fig = px.bar(
            reviews_by_game.sort_values("review_count"),
            x="review_count",
            y="title",
            orientation="h",
            title="Raw Review Count Imbalance",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)


with features_tab:
    st.subheader("Transformed Game-Level Dataset")
    st.markdown(
        '<div class="small-note">The dashboard converts review-level data into game-level '
        "features because GameVault makes acquisition decisions about games, not individual reviews.</div>",
        unsafe_allow_html=True,
    )

    feature_cols = [
        "title",
        "review_count",
        "review_share",
        "recommended_proportion",
        "average_hours_played",
        "median_hours_played",
        "review_span_days",
        "helpful_vote_share",
        "helpful_review_proportion",
        "average_helpful_per_review",
        "game_sentiment_score",
        "sentiment_positive_share",
        "sentiment_negative_share",
        "early_access_review_share",
    ]
    st.dataframe(
        game_df[feature_cols].style.format(
            {
                "review_share": "{:.1%}",
                "recommended_proportion": "{:.1%}",
                "average_hours_played": "{:.1f}",
                "median_hours_played": "{:.1f}",
                "helpful_vote_share": "{:.1%}",
                "helpful_review_proportion": "{:.1%}",
                "average_helpful_per_review": "{:.2f}",
                "game_sentiment_score": "{:.3f}",
                "sentiment_positive_share": "{:.1%}",
                "sentiment_negative_share": "{:.1%}",
                "early_access_review_share": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    heatmap_cols = {
        "review_count": "Review Volume",
        "recommended_proportion": "Recommendation Rate",
        "median_hours_played": "Median Playtime",
        "review_span_days": "Review Longevity",
        "helpful_review_proportion": "Helpful Review Share",
        "game_sentiment_score": "Constructed Sentiment",
    }
    heatmap_data = game_df[["title"] + list(heatmap_cols)].copy()
    heatmap_data["review_count"] = np.log1p(heatmap_data["review_count"])
    heatmap_data["median_hours_played"] = np.log1p(heatmap_data["median_hours_played"])
    scaled = heatmap_data[list(heatmap_cols)].apply(minmax)
    scaled.columns = list(heatmap_cols.values())
    scaled.insert(0, "title", heatmap_data["title"])

    fig = px.imshow(
        scaled.drop(columns="title"),
        x=scaled.drop(columns="title").columns,
        y=scaled["title"],
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Game Feature Heatmap",
        labels={"x": "Feature", "y": "Game", "color": "Scaled value"},
    )
    fig.update_layout(height=760, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


with eda_tab:
    st.subheader("EDA Evidence Behind Transformations")
    st.markdown(
        '<div class="small-note">This section mirrors the notebook EDA: first inspect '
        "coverage and game-level relationships, then inspect skewness to justify scoring transformations.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Review Coverage and Candidate Signals")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            game_df.sort_values("review_share", ascending=True),
            x="review_share",
            y="title",
            orientation="h",
            title="Total Review Proportion by Game",
            hover_data=["review_count", "recommended_proportion"],
            labels={"review_share": "Share of all reviews", "title": ""},
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(
            game_df,
            x="review_count",
            y="recommended_proportion",
            size="median_hours_played",
            color="game_sentiment_score",
            hover_name="title",
            log_x=True,
            title="Recommendation Rate vs Review Volume",
            labels={
                "review_count": "Review count, log axis",
                "recommended_proportion": "Recommendation rate",
                "game_sentiment_score": "Sentiment score",
            },
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="small-note">Bubble size represents median hours played. Colour represents '
            "constructed sentiment score. The x-axis is log-scaled review count.</div>",
            unsafe_allow_html=True,
        )

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Recommendation and Sentiment Alignment")
        sentiment_alignment = game_df[
            [
                "title",
                "recommended_proportion",
                "game_sentiment_score",
                "sentiment_positive_share",
                "sentiment_negative_share",
                "sentiment_neutral_share",
                "sentiment_confidence",
                "review_count",
            ]
        ].copy()
        sentiment_alignment["approval_sentiment_gap"] = (
            sentiment_alignment["recommended_proportion"]
            - sentiment_alignment["game_sentiment_score"]
        )
        sentiment_alignment = sentiment_alignment.sort_values(
            ["recommended_proportion", "game_sentiment_score"],
            ascending=False,
        )
        st.dataframe(
            sentiment_alignment.style.format(
                {
                    "recommended_proportion": "{:.1%}",
                    "game_sentiment_score": "{:.3f}",
                    "sentiment_positive_share": "{:.1%}",
                    "sentiment_negative_share": "{:.1%}",
                    "sentiment_neutral_share": "{:.1%}",
                    "sentiment_confidence": "{:.3f}",
                    "approval_sentiment_gap": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            '<div class="small-note">This table is used as a sentiment sanity check. '
            "Recommendation is the player's explicit judgement; sentiment is supporting "
            "review-text context and should not replace recommendation rate.</div>",
            unsafe_allow_html=True,
        )

    with col_d:
        fig = px.scatter(
            game_df,
            x="median_hours_played",
            y="game_sentiment_score",
            size="review_count",
            color="recommended_proportion",
            hover_name="title",
            log_x=True,
            title="Player Engagement vs Sentiment",
            labels={
                "median_hours_played": "Median hours played, log axis",
                "game_sentiment_score": "Constructed sentiment score",
                "recommended_proportion": "Recommendation rate",
            },
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="small-note">Bubble size represents review count. Colour represents '
            "recommendation rate. The x-axis is log-scaled median playtime.</div>",
            unsafe_allow_html=True,
        )

    fig = px.scatter(
        game_df,
        x="review_span_days",
        y="review_count",
        color="recommended_proportion",
        size="game_sentiment_score",
        hover_name="title",
        title="Review Longevity vs Review Volume",
        labels={
            "review_span_days": "Review span in days",
            "review_count": "Review count",
            "recommended_proportion": "Recommendation rate",
        },
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="small-note">Bubble size represents constructed sentiment score. Colour represents '
        "recommendation rate. This compares sustained review activity against review volume.</div>",
        unsafe_allow_html=True,
    )

    dist_cols = [
        "review_count",
        "review_share",
        "recommended_proportion",
        "average_hours_played",
        "median_hours_played",
        "review_span_days",
        "helpful_vote_share",
        "helpful_review_proportion",
        "average_helpful_per_review",
        "game_sentiment_score",
    ]
    summary = game_df[dist_cols].describe().T
    summary["skew"] = game_df[dist_cols].skew()
    st.dataframe(summary.round(3), use_container_width=True)

    helpful_weight_check = raw_df[["helpful"]].copy()
    helpful_weight_check["log1p_helpful"] = np.log1p(helpful_weight_check["helpful"])
    helpful_weight_long = helpful_weight_check.melt(
        value_vars=["helpful", "log1p_helpful"],
        var_name="feature",
        value_name="value",
    )
    fig = px.box(
        helpful_weight_long,
        x="feature",
        y="value",
        points="outliers",
        title="Review-Level Helpful Votes Before and After log1p Transformation",
    )
    st.plotly_chart(fig, use_container_width=True)

    kde_df = build_kde_frame(game_df, dist_cols)
    fig = px.line(
        kde_df,
        x="value",
        y="density",
        facet_col="feature",
        facet_col_wrap=3,
        title="Smoothed Game-Level Feature Distributions",
    )
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None, showticklabels=False)
    fig.update_layout(height=860, showlegend=False)
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)

    scaling_plan = pd.DataFrame(
        [
            ["review_count", "log1p then rank scale", "Skewed count; rank scaling is robust to outliers."],
            ["median_hours_played", "log1p then rank scale", "Unbounded engagement duration; compared relatively across games."],
            ["average_helpful_per_review", "log1p then rank scale", "Helpful intensity is highly skewed."],
            ["review_span_days", "rank scale", "Magnitude feature compared relatively across games."],
            ["recommended_proportion", "use directly", "Already a meaningful 0-1 player approval rate."],
            ["helpful_review_proportion", "use directly", "Already a meaningful 0-1 proportion."],
            ["game_sentiment_score", "use directly", "Constructed bounded sentiment score, supporting signal."],
        ],
        columns=["Feature", "Scoring transformation", "Reason"],
    )
    st.subheader("Scaling Decisions Used Later in Scoring")
    st.dataframe(scaling_plan, use_container_width=True, hide_index=True)

    log_features = game_df.copy()
    log_cols = ["review_count", "median_hours_played", "average_helpful_per_review"]
    for col in log_cols:
        log_features[f"log1p_{col}"] = np.log1p(log_features[col])
    log_kde = build_kde_frame(log_features, [f"log1p_{col}" for col in log_cols], value_column="log1p_value")
    log_kde["feature"] = log_kde["feature"].str.replace("log1p_", "", regex=False)

    fig = px.line(
        log_kde,
        x="log1p_value",
        y="density",
        facet_col="feature",
        facet_col_wrap=3,
        title="Log-Scaled Distributions for Selected Scoring Features",
    )
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None, showticklabels=False)
    fig.update_layout(height=460, showlegend=False)
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)


with sentiment_tab:
    st.subheader("Sentiment Construction and Validation")

    review_df["review_word_count"] = review_df["review"].fillna("").str.findall(r"[A-Za-z']+").str.len()
    word_summary = review_df["review_word_count"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    col_a, col_b = st.columns([0.75, 1.25])
    with col_a:
        st.metric("Median review length", f"{word_summary['50%']:.0f} words")
        st.metric("95th percentile length", f"{word_summary['95%']:.0f} words")
        st.metric("Maximum review length", f"{word_summary['max']:.0f} words")
    with col_b:
        st.markdown(
            '<div class="decision-box">VADER is used because most reviews are short, informal '
            "player comments. It provides negative, neutral, positive and compound scores. "
            "Blank reviews are kept as neutral so non-text signals are not discarded.</div>",
            unsafe_allow_html=True,
        )

    sentiment_table = pd.crosstab(
        review_df["recommendation"],
        review_df["sentiment_label"],
        normalize="index",
    ).round(3)
    st.subheader("Does Sentiment Agree With Steam Recommendation?")
    st.dataframe(sentiment_table, use_container_width=True)

    fig = px.histogram(
        review_df,
        x="sentiment_score",
        color="recommendation",
        barmode="overlay",
        nbins=50,
        title="VADER Compound Sentiment by Recommendation",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="small-note">VADER separates recommended reviews well, but is weaker on '
        "not-recommended reviews because negative recommendations often contain mixed positive wording. "
        "Therefore sentiment supports approval; it does not replace the recommendation label.</div>",
        unsafe_allow_html=True,
    )

    mismatch = review_df[
        review_df["recommendation"].eq("Not Recommended")
        & review_df["sentiment_label"].eq("positive")
    ].sort_values("sentiment_score", ascending=False)
    st.dataframe(
        mismatch[["title", "sentiment_score", "hour_played", "helpful", "review"]].head(20),
        use_container_width=True,
        hide_index=True,
    )


with scoring_tab:
    st.subheader("Transparent Acquisition Scoring")

    st.markdown(
        """
        The final score is a weighted prioritisation model, not a black-box prediction.

        `final_score = 35% approval + 25% engagement + 20% evidence + 15% longevity + 5% review quality`

        """
    )

    score_definitions = pd.DataFrame(
        [
            ["Approval", "85% recommendation rate + 15% sentiment", "35%", "Do players broadly endorse the game?"],
            ["Engagement", "rank-scaled log median playtime", "25%", "Do players spend meaningful time in the game?"],
            ["Evidence", "rank-scaled log review count", "20%", "Is there enough review volume to trust the signal?"],
            ["Longevity", "rank-scaled review span days", "15%", "Is interest sustained over time?"],
            ["Review quality", "helpful proportion + rank-scaled helpful intensity", "5%", "Are reviews useful to other players?"],
        ],
        columns=["Sub-score", "Feature basis", "Weight", "Client question"],
    )
    st.dataframe(score_definitions, use_container_width=True, hide_index=True)

    st.subheader("Risk Flags")
    risk_definitions = pd.DataFrame(
        [
            ["Low review count", "review_count < 25", "Limited review evidence; score may be less reliable."],
            ["Short review span", "review_span_days < 90", "Potential short-term buzz rather than sustained demand."],
            ["Mixed approval", "recommended_proportion < 0.70", "Player approval is mixed; qualitative review is needed."],
        ],
        columns=["Risk flag", "Rule", "Decision meaning"],
    )
    st.dataframe(risk_definitions, use_container_width=True, hide_index=True)
 

    st.dataframe(
        game_df[
            [
                "opportunity_rank",
                "title",
                "opportunity_tier",
                "final_score",
                "approval_score",
                "engagement_score",
                "evidence_score",
                "longevity_score",
                "review_quality_score",
                "risk_flag_low_review_count",
                "risk_flag_short_review_span",
                "risk_flag_mixed_approval",
                "risk_flag_count",
            ]
        ].style.format(
            {
                "final_score": "{:.3f}",
                "approval_score": "{:.3f}",
                "engagement_score": "{:.3f}",
                "evidence_score": "{:.3f}",
                "longevity_score": "{:.3f}",
                "review_quality_score": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    components = ranking_component_long(game_df, top_n=top_n)
    top_titles = game_df.head(top_n)["title"].tolist()
    fig = px.bar(
        components,
        x="component_value",
        y="title",
        color="score_component",
        orientation="h",
        title="Why the Top Games Score Highly",
        category_orders={"title": list(reversed(top_titles))},
    )
    fig.update_layout(height=650, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


with business_tab:
    st.subheader("Business Value and Decision Recommendations")

    selected_game = st.selectbox(
        "Select game for drill-down",
        game_df["title"].tolist(),
        index=0,
    )

    selected_row = game_df.loc[game_df["title"].eq(selected_game)].iloc[0]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rank", int(selected_row["opportunity_rank"]))
    col_b.metric("Final score", f"{selected_row['final_score']:.3f}")
    col_c.metric("Tier", selected_row["opportunity_tier"])
    col_d.metric("Risk flags", int(selected_row["risk_flag_count"]))

    decision_cols = st.columns([1, 1])
    with decision_cols[0]:
        st.markdown(
            f"""
            <div class="decision-box">
            <strong>Decision interpretation for {selected_game}</strong><br>
            Approval: {selected_row['approval_score']:.3f}<br>
            Engagement: {selected_row['engagement_score']:.3f}<br>
            Evidence: {selected_row['evidence_score']:.3f}<br>
            Longevity: {selected_row['longevity_score']:.3f}<br>
            Review quality: {selected_row['review_quality_score']:.3f}<br>
            Risk context: {risk_text(selected_row)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with decision_cols[1]:
        recommendation = (
            "Prioritise for commercial due diligence"
            if selected_row["opportunity_tier"] == "Priority Target"
            else "Keep in shortlist and validate externally"
            if selected_row["opportunity_tier"] == "Strong Candidate"
            else "Monitor or investigate only if strategically aligned"
        )
        st.markdown(
            f"""
            <div class="decision-box">
            <strong>Recommended action</strong><br>
            {recommendation}. Next checks should include developer fit, acquisition cost,
            platform availability, revenue estimates, genre strategy, and recent market traction.
            </div>
            """,
            unsafe_allow_html=True,
        )

    selected_reviews = review_df[review_df["title"].eq(selected_game)].copy()
    selected_reviews = selected_reviews.sort_values(["helpful", "sentiment_score"], ascending=False)

    fig = px.scatter(
        selected_reviews,
        x="hour_played",
        y="sentiment_score",
        color="recommendation",
        size="helpful",
        hover_data=["sentiment_label", "date_posted"],
        title=f"{selected_game}: Review Sentiment vs Player Hours",
        labels={"hour_played": "Hours played", "sentiment_score": "VADER compound sentiment"},
    )
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="small-note">Bubble size represents helpful votes for the individual review. '
        "Colour represents the Steam recommendation label.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Review Drill-Down")
    st.dataframe(
        selected_reviews[
            [
                "date_posted",
                "recommendation",
                "hour_played",
                "helpful",
                "sentiment_score",
                "sentiment_label",
                "review",
            ]
        ].head(50),
        use_container_width=True,
        hide_index=True,
    )
