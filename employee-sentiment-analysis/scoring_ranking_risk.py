# scoring_ranking_risk.py
# Tasks 3, 4, 5: Monthly scoring, ranking, and flight risk detection

import pandas as pd
from datetime import timedelta
from collections import defaultdict


def calculate_monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 3: Compute monthly sentiment score for each employee
    Positive = +1, Negative = -1, Neutral = 0
    Score resets each month
    """
    print("\n" + "=" * 70)
    print("TASK 3: MONTHLY SENTIMENT SCORE CALCULATION")
    print("=" * 70)

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create year-month column
    df['year_month'] = df['date'].dt.to_period('M')

    # Map sentiment to score
    score_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['score'] = df['sentiment'].map(score_map)

    # Group by employee and month, sum scores
    monthly_scores = df.groupby(['employee_email', 'year_month'])['score'].sum().reset_index()
    monthly_scores = monthly_scores.rename(columns={'score': 'monthly_sentiment_score'})

    print("Monthly scores calculated for each employee.")
    print(f"Total employee-month records: {len(monthly_scores)}")
    print("\nSample monthly scores:")
    print(monthly_scores.sort_values(['year_month', 'monthly_sentiment_score'], ascending=[True, False]).head(10))

    return monthly_scores


def rank_employees(monthly_scores: pd.DataFrame):
    """
    Task 4: Top 3 positive and negative employees per month
    Sorted by score descending, then alphabetically
    """
    print("\n" + "=" * 70)
    print("TASK 4: EMPLOYEE RANKING (Top 3 Positive & Negative per Month)")
    print("=" * 70)

    ranking_results = []

    for month in monthly_scores['year_month'].unique():
        month_data = monthly_scores[monthly_scores['year_month'] == month].copy()

        # Sort: highest score first, then alphabetical
        month_data = month_data.sort_values(
            ['monthly_sentiment_score', 'employee_email'],
            ascending=[False, True]
        )

        # Top 3 positive
        top_positive = month_data.head(3)[['employee_email', 'monthly_sentiment_score']].values.tolist()

        # Top 3 negative (lowest scores)
        bottom_negative = month_data.tail(3)[['employee_email', 'monthly_sentiment_score']].values.tolist()
        # Reverse to show most negative first
        bottom_negative = sorted(bottom_negative, key=lambda x: x[1])

        print(f"\nMonth: {month}")
        print("Top 3 Positive:")
        for i, (emp, score) in enumerate(top_positive, 1):
            print(f"  {i}. {emp} (Score: {score})")

        print("Top 3 Negative:")
        for i, (emp, score) in enumerate(bottom_negative, 1):
            print(f"  {i}. {emp} (Score: {score})")

        ranking_results.append({
            'month': month,
            'top_positive': top_positive,
            'top_negative': bottom_negative
        })

    return ranking_results


def identify_flight_risks(df: pd.DataFrame) -> list:
    """
    Task 5: Flight risk = ≥4 negative messages in any rolling 30-day window
    """
    print("\n" + "=" * 70)
    print("TASK 5: FLIGHT RISK IDENTIFICATION")
    print("=" * 70)

    df['date'] = pd.to_datetime(df['date'])
    df_negative = df[df['sentiment'] == 'Negative'].copy()
    df_negative = df_negative.sort_values(['employee_email', 'date'])

    flight_risk_employees = set()

    for employee in df_negative['employee_email'].unique():
        emp_dates = df_negative[df_negative['employee_email'] == employee]['date'].tolist()

        # Check every 30-day rolling window
        for i in range(len(emp_dates)):
            window_start = emp_dates[i]
            window_end = window_start + timedelta(days=30)
            count_in_window = sum(1 for d in emp_dates[i:] if d <= window_end)
            if count_in_window >= 4:
                flight_risk_employees.add(employee)
                break  # No need to check further for this employee

    print(f"Employees flagged as flight risk (≥4 negative messages in 30 days): {len(flight_risk_employees)}")
    if flight_risk_employees:
        print("List:")
        for emp in sorted(flight_risk_employees):
            print(f"  • {emp}")
    else:
        print("  None")

    return sorted(list(flight_risk_employees))