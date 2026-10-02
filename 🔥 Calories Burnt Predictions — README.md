🔥 CaloriesBurnt [Overview](#overview) [Dataset](#dataset) [Analysis](#analysis) [Model](#model) [Demo](#demo) [Resume](#resume) [Setup](#setup) [⭐ GitHub](https://github.com/Malukchand/Calories-Burnt-Predictions-)

📊 Linear Regression · May'25 – Jun'25 · Kaggle Dataset

# 🔥 Calories Burnt Predictions

A statistical machine learning project that identifies the key physiological drivers of calorie burn — using **Linear Regression**, IQR-based outlier detection, and rigorous residual diagnostics per feature.

[📓 Open Notebook](https://github.com/Malukchand/Calories-Burnt-Predictions-/blob/main/Calories_Burnt_Predictions.ipynb) [📈 See Analysis](#analysis)

0.96R² · Duration

0.90R² · Heart Rate

0.82R² · Body Temp

3Features Modelled

LinearRegression

---

Project Overview

## Per-feature statistical analysis of calorie expenditure

Rather than a black-box ensemble, this project takes a rigorous statistical approach: each feature is individually examined for normality, outliers, linearity, and model fit before a final multi-feature regression. The result is a fully interpretable model with explained variance.

🧹

Data Cleaning

Dropped User_ID, mapped Gender to binary (male=1, female=0), checked null values — none found.

📊

EDA & Visualisation

Correlation heatmap, boxplots, histograms, and count plots to understand distributions and feature relationships.

🎯

Outlier Detection

IQR-based outlier function applied to Height, Weight, Heart Rate, Body Temp, and Calories.

📉

Normality Testing

Frequency + density histograms and Normal Q-Q plots for each predictor variable.

📐

Linearity Checks

Scatter plots, average-calories bar plots, and regression line fits per feature before modelling.

🔬

Residual Diagnostics

4-panel diagnostic plots per feature: Residuals vs Fitted, Q-Q Residuals, Scale-Location, Residuals vs Leverage.

---

Data

## Kaggle exercise dataset — 7 features, 1 target

Two files — `exercise.csv` and `calories.csv` — merged on User_ID. The ID column is then dropped as it carries no predictive information.

👤

Gender

binary encoded · male=1 female=0

🎂

Age

numeric · years

📏

Height

numeric · cm · outliers checked

⚖️

Weight

numeric · kg · outliers checked

⏱️

Duration

numeric · min · R²=0.96 ★ top feature

💓

Heart Rate

numeric · bpm · R²=0.90

🌡️

Body Temp

numeric · °C · R²=0.82

🔥

Calories

target · kcal · positively skewed

---

Per-Feature Analysis

## Each feature gets the full statistical treatment

The notebook runs an identical 6-step analysis pipeline for every key predictor — click a tab to see what was done and why it matters.

⏱️

**Duration of Exercise**

The strongest predictor of calorie burn — near-perfect linear relationship

R² = 0.96

R² Score

0.96

Relationship

Strong linear

Outliers Found

None

Distribution

Near-normal

Explanatory Power96%

1

Frequency + Density Histograms

Both histograms confirm Duration is approximately normally distributed.

2

Normal Q-Q Plot

Points closely follow the reference line → assumption of normality satisfied.

3

Scatter Plot (Duration vs Calories)

Clear, strong positive linear trend — longer workout = proportionally more calories.

4

Bar Plot: Avg Calories per Duration Group

Binned groups confirm the monotonically increasing calorie trend with duration.

5

Linear Regression Fit (sklearn)

*R² = 0.96, very low RMSE* — Duration alone explains 96% of calorie variance.

6

4-Panel Residual Diagnostics (statsmodels)

Residuals vs Fitted, Q-Q Residuals, Scale-Location, and Residuals vs Leverage confirm homoscedasticity and no influential outliers.

💓

**Heart Rate**

Second strongest predictor — physiologically intuitive

R² = 0.90

R² Score

0.90

Relationship

Strong linear

Outliers Found

Checked via IQR

Distribution

Near-normal

Explanatory Power90%

1

Frequency + Density Histograms

Heart Rate distribution is roughly symmetric and bell-shaped.

2

Normal Q-Q Plot

Light tails but broadly normal — linearity assumption holds.

3

Scatter Plot (Heart Rate vs Calories)

Positive linear trend: higher BPM during exercise → more calories burned.

4

Bar Plot: Avg Calories per Heart Rate Group

Binned groups show clean step-up in average calories as heart rate increases.

5

Linear Regression Fit

*R² = 0.90* — Heart Rate explains 90% of calorie variance independently.

6

4-Panel Residual Diagnostics

Moderate residual spread at higher fitted values — some heteroscedasticity, consistent with more variability at high effort levels.

🌡️

**Body Temperature**

Third key predictor — thermogenesis and metabolic rate indicator

R² = 0.82

R² Score

0.82

Relationship

Good linear

Outliers Found

Checked via IQR

Distribution

Narrow range

Explanatory Power82%

1

Frequency + Density Histograms

Body Temp ranges narrowly (36–42°C) with a near-normal distribution.

2

Normal Q-Q Plot

Q-Q confirms approximate normality within the physiological range.

3

Scatter Plot (Body Temp vs Calories)

Positive trend — higher body temp during exercise correlates with more calories burned.

4

Bar Plot: Avg Calories per Temp Group

Shows a clear upward staircase pattern in average calories as temperature rises.

5

Linear Regression Fit

*R² = 0.82* — Body Temp explains 82% of calorie variance on its own.

6

4-Panel Residual Diagnostics

Residuals are acceptably uniform across the narrow temperature range.

📏

**Height**

Weaker predictor — analysed to demonstrate EDA methodology

Low R²

Outliers Found

Yes — IQR flagged

Relationship

Weak linear

Normal Q-Q

Approximately normal

Takeaway

Not a top driver

1

Outlier Detection via IQR

The `get_outliers()` function flagged some extreme height values using Q1 − 1.5×IQR and Q3 + 1.5×IQR bounds.

2

Frequency + Density Histograms

Bimodal shape visible — reflects gender-based height distribution in the dataset.

3

Normal Q-Q Plot

Slight deviations at tails, consistent with the bimodal pattern.

4

Linear Regression + Diagnostics

Low R² confirms height alone is not a strong calorie predictor — but the diagnostic pipeline is identical, demonstrating rigour.

---

Model

## Linear Regression — interpretable and effective

The project uses scikit-learn's `LinearRegression` and statsmodels' `OLS` per feature, with MAE, MSE, and R² as evaluation metrics.

Python — per-feature regression pipeline

from sklearn.linear_model import LinearRegression from sklearn.metrics import mean_squared_error, r2_score import statsmodels.api as sm # ── Per-feature sklearn fit ────────────────────── X = df\[\['Duration'\]\] # swap in Heart_Rate, Body_Temp, etc. y = df\['Calories'\] model = LinearRegression() model.fit(X, y) y_pred = model.predict(X) r2 = r2_score(y, y_pred) # Duration → 0.96 rmse = np.sqrt(mean_squared_error(y, y_pred)) print(f"R²: {r2:.2f} | RMSE: {rmse:.2f}") # ── IQR outlier detection ──────────────────────── def get_outliers(df, col): q1, q3 = df\[col\].quantile(\[.25, .75\]) iqr = q3 - q1 return df\[col\]\[(df\[col\] \< q1 - 1.5*iqr) | (df\[col\] > q3 + 1.5*iqr)\]

---

Interactive Demo

## Estimate your calorie burn

Uses the linear regression coefficients from the notebook. Duration has the highest weight (R²=0.96), followed by Heart Rate and Body Temp.

🔥 Calorie Burn Estimator

Approximate prediction using learned linear weights from the notebook's models

Gender

Age

25 yrs

Weight (kg)

70 kg

Height (cm)

175 cm

Duration (min) — top feature

30 min

Heart Rate (bpm)

110 bpm

Body Temp (°C)

38.5 °C

0

kcal burned

Roughly equivalent to:

Estimated via linear weights: Duration (coeff \~7.2) + Heart Rate + Body Temp

---

Resume Format

## Ready to paste into your CV

Exactly how this project appears in your resume — just click copy.

📄 Calories Burnt Predictions | Self Project | ⭐ (May'25 – Jun'25)

- Built a calorie expenditure prediction model using user activity, health, and demographic features from a Kaggle dataset
- Cleaned and analyzed the dataset using Pandas, visualizing trend patterns with histograms, boxplots and correlation heatmaps
- Trained and evaluated a Linear Regression model using scikit-learn with R-squared, MAE and MSE metrics
- Identified activity duration (0.96), heart rate (0.90), body temperature (0.82) as the strongest major factors influencing calorie burn

---

Setup

## Run it in 3 commands

🐍 Python 3.8+

📓 Jupyter / Colab

🐼 Pandas

🔢 NumPy

📊 Matplotlib

🌊 Seaborn

⚙️ scikit-learn

📐 statsmodels

🧮 scipy

bash

\# 1. Clone git clone https://github.com/Malukchand/Calories-Burnt-Predictions-.git cd Calories-Burnt-Predictions- # 2. Install dependencies pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy jupyter # 3. Run jupyter notebook Calories_Burnt_Predictions.ipynb

---

Maluk — BSBE, IIT Kanpur

3rd Year · Biological Sciences & Bioengineering · Batch 2023 · May–Jun 2025

[🐙 GitHub](https://github.com/Malukchand) [📁 Repo](https://github.com/Malukchand/Calories-Burnt-Predictions-) [📓 Notebook](https://github.com/Malukchand/Calories-Burnt-Predictions-/blob/main/Calories_Burnt_Predictions.ipynb)

MIT License · Dataset from Kaggle · Built May–Jun 2025