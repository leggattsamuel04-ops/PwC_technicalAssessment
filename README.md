# PwC Technical Assessment

This project analyses Steam review data for the GameVault acquisition scenario. The aim is to turn individual player reviews into a simple game-level ranking that helps identify promising acquisition targets.

## What This Project Does

- Loads the original review-level Steam dataset.
- Explores review imbalance, recommendation rates, playtime, helpful votes, and feature skewness.
- Builds a transformed game-level dataset with one row per game.
- Uses VADER sentiment analysis to add review text sentiment.
- Creates a transparent acquisition score using approval, engagement, evidence, longevity, and review quality.
- Presents the results in a Streamlit dashboard for demonstration.

## Approach

The original data is at review level, but the business decision is at game level. I therefore aggregated the reviews by game and created features that describe each game's commercial opportunity:

- Approval: proportion of reviews that recommend the game, supported by sentiment.
- Engagement: median hours played, showing how deeply players interact with the game.
- Evidence: number of reviews, showing how much data supports the signal.
- Longevity: review span over time, showing whether interest is sustained.
- Review quality: helpful review activity, showing whether reviews contain useful buyer information.

Skewed count-based features, such as review count and playtime, were log transformed and rank scaled before scoring. Proportions, such as recommendation rate, were kept directly because they already have a clear 0-1 interpretation.

## Key Findings

- The raw dataset is imbalanced towards recommended reviews, so recommendation rate alone should not decide acquisition priority.
- Review counts are uneven across games, meaning high-volume games provide stronger evidence than low-volume games.
- VADER sentiment broadly agrees with recommendations, but it is weaker at identifying negative non-recommended reviews, so sentiment is used as a supporting signal rather than the main approval measure.
- Some games show strong approval but lower evidence or shorter review history, which creates acquisition risk.
- The strongest candidates combine high approval, strong playtime engagement, enough review evidence, and sustained review activity over time.

## Recommendation

GameVault should prioritise games with high final acquisition scores and low risk flags. The shortlist should be used as a first-stage screening tool, not a final purchase decision. For top-ranked games, the next step would be commercial due diligence, including price, ownership, genre fit, technical condition, and expected return on acquisition.

## How To Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Main Files

- `task.ipynb`: Main notebook used for EDA, sentiment analysis, feature engineering, and scoring.
- `app.py`: Streamlit dashboard used to present the analysis.
- `steam_reviews 1.csv`: Original review-level dataset.
- `game_level_features.csv`: Final transformed game-level dataset.
- `review_level_sentiment.csv`: Review-level dataset with sentiment scores.
- `game_reviews_table.json`: Review drill-down data by game.

## Dashboard Structure

The Streamlit app follows the project story:

1. Raw review data
2. Game-level transformed data
3. EDA and scaling decisions
4. Acquisition shortlist
5. Sentiment analysis
6. Scoring method
7. Business recommendations

## AI Use

ChatGPT was used as a coding and writing assistant during this project. In particular, it helped write and structure the Streamlit website in `app.py`, improve explanations, and support iterative changes to the dashboard. The analysis decisions, scoring logic, and final interpretation were reviewed and adapted for this assessment.
