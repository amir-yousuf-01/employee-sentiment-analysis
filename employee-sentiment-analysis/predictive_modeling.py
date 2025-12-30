# predictive_modeling.py
# Task 6: Linear Regression to predict monthly sentiment score

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

VIS_FOLDER = Path("visualizations")

def predictive_modeling(df: pd.DataFrame, monthly_scores: pd.DataFrame):
    """
    Task 6: Linear regression model to predict monthly sentiment score
    Features: message count, avg length, etc.
    """
    print("\n" + "="*70)
    print("TASK 6: PREDICTIVE MODELING (LINEAR REGRESSION)")
    print("="*70)

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')

    # Feature engineering per employee-month
    features = df.groupby(['employee_email', 'year_month']).agg(
        message_count=('body', 'count'),
        avg_length=('body', lambda x: x.str.len().mean()),
        total_words=('body', lambda x: x.str.split().str.len().sum()),
        negative_count=('sentiment', lambda x: (x == 'Negative').sum()),
        positive_count=('sentiment', lambda x: (x == 'Positive').sum())
    ).reset_index()

    # Merge with target
    data = features.merge(monthly_scores, on=['employee_email', 'year_month'], how='left')
    data['monthly_sentiment_score'] = data['monthly_sentiment_score'].fillna(0)

    print(f"Prepared {len(data)} employee-month records for modeling")
    print("\nSample features:")
    print(data.head())

    # Features and target
    X = data[['message_count', 'avg_length', 'total_words', 'negative_count', 'positive_count']]
    y = data['monthly_sentiment_score']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Coefficients
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    print("\nFeature Coefficients:")
    print(coef_df)

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Monthly Score")
    plt.ylabel("Predicted Monthly Score")
    plt.title(f"Actual vs Predicted Monthly Sentiment Score (R² = {r2:.3f})")  # Fixed line
    plt.tight_layout()
    plt.savefig(VIS_FOLDER / "predictive_model_actual_vs_predicted.png", dpi=200)
    plt.close()
    print("→ Saved: visualizations/predictive_model_actual_vs_predicted.png")

    # Plot 2: Feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(coef_df['feature'], coef_df['coefficient'], color='skyblue')
    plt.xlabel("Coefficient Value")
    plt.title("Feature Importance in Linear Regression Model")
    plt.tight_layout()
    plt.savefig(VIS_FOLDER / "predictive_model_feature_importance.png", dpi=200)
    plt.close()
    print("→ Saved: visualizations/predictive_model_feature_importance.png")

    print("\nTask 6 completed!")