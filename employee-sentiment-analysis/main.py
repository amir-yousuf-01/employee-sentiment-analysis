# main.py
# Final main script - All Tasks 1–6

from utils import load_data, save_processed_data
from sentiment_labeling import add_sentiment_column
from eda import perform_eda
from scoring_ranking_risk import calculate_monthly_scores, rank_employees, identify_flight_risks
from predictive_modeling import predictive_modeling

if __name__ == "__main__":
    print("=" * 80)
    print("     EMPLOYEE SENTIMENT ANALYSIS PROJECT - FULL RUN")
    print("=" * 80)

    df = load_data()
    df = add_sentiment_column(df)
    save_processed_data(df, "processed/labeled_data.csv")

    perform_eda(df)

    monthly_scores = calculate_monthly_scores(df)
    rank_employees(monthly_scores)
    flight_risks = identify_flight_risks(df)

    predictive_modeling(df, monthly_scores)

    with open("flight_risks.txt", "w") as f:
        f.write("\n".join(flight_risks))

    print("\n" + "="*80)
    print("ALL TASKS 1–6 COMPLETED SUCCESSFULLY!")
    print("Project ready for submission!")
    print("="*80)