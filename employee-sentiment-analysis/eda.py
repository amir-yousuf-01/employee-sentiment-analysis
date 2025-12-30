# eda.py
# Task 2: Exploratory Data Analysis (EDA) with visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create visualizations folder if it doesn't exist
VIS_FOLDER = Path("visualizations")
VIS_FOLDER.mkdir(parents=True, exist_ok=True)  # Fixed line

def perform_eda(df: pd.DataFrame):
    """
    Perform Exploratory Data Analysis and save visualizations.
    """
    print("\n" + "="*70)
    print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    # 1. Data structure
    print("\n1. Data Structure:")
    print(f"Total messages: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # 2. Sentiment distribution
    print("\n2. Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    print("\nPercentage:")
    print((sentiment_counts / len(df) * 100).round(2))

    # Visualization 1: Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=['salmon', 'lightgreen', 'lightgray'])
    plt.title("Overall Sentiment Distribution")
    plt.savefig(VIS_FOLDER / "sentiment_distribution_pie.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("→ Saved: visualizations/sentiment_distribution_pie.png")

    # 3. Messages over time
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_counts = df['year_month'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    monthly_counts.plot(kind='line', marker='o', color='steelblue')
    plt.title("Number of Messages Over Time (Monthly)")
    plt.xlabel("Year-Month")
    plt.ylabel("Message Count")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIS_FOLDER / "messages_over_time.png", dpi=200)
    plt.close()
    print("→ Saved: visualizations/messages_over_time.png")

    # 4. Sentiment over time
    monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    monthly_sentiment = monthly_sentiment.reindex(monthly_counts.index, fill_value=0)

    plt.figure(figsize=(12, 6))
    monthly_sentiment.plot(kind='bar', stacked=True,
                           color={'Positive': 'lightgreen', 'Negative': 'salmon', 'Neutral': 'lightgray'})
    plt.title("Sentiment Trends Over Time (Monthly)")
    plt.xlabel("Year-Month")
    plt.ylabel("Message Count")
    plt.legend(title="Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIS_FOLDER / "sentiment_over_time.png", dpi=200)
    plt.close()
    print("→ Saved: visualizations/sentiment_over_time.png")

    # 5. Messages per employee
    employee_counts = df['employee_email'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(y=employee_counts.index, x=employee_counts.values, palette='viridis')
    plt.title("Messages per Employee")
    plt.xlabel("Number of Messages")
    plt.ylabel("Employee Email")
    plt.tight_layout()
    plt.savefig(VIS_FOLDER / "messages_per_employee.png", dpi=200)
    plt.close()
    print("→ Saved: visualizations/messages_per_employee.png")

    # 6. Message length analysis
    df['msg_length'] = df['body'].astype(str).str.len()

    print("\n6. Message Length Insights:")
    print(f"Average length: {df['msg_length'].mean():.0f} characters")
    print(f"Median length: {df['msg_length'].median():.0f} characters")
    print(f"Longest message: {df['msg_length'].max()} characters")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment', y='msg_length', data=df,
                palette={'Positive': 'lightgreen', 'Negative': 'salmon', 'Neutral': 'lightgray'})
    plt.title("Message Length by Sentiment")
    plt.ylabel("Message Length (characters)")
    plt.ylim(0, df['msg_length'].quantile(0.95))  # Remove extreme outliers
    plt.tight_layout()
    plt.savefig(VIS_FOLDER / "message_length_by_sentiment.png", dpi=200)
    plt.close()
    print("→ Saved: visualizations/message_length_by_sentiment.png")

    print("\n" + "="*70)
    print("TASK 2 COMPLETED! All visualizations saved in 'visualizations/' folder")
    print("="*70)