# Employee Sentiment Analysis Project

## Summary
This project analyzes employee email messages to assess sentiment, engagement, and flight risk using NLP and statistical methods.

**Key Findings:**
- Total messages analyzed: 2191 from 10 employees (2010–2011)
- Overall sentiment: 57.69% Negative, 37.20% Positive, 5.11% Neutral
- All 10 employees flagged as **flight risk** (≥4 negative messages in a 30-day window)
- Predictive model achieved perfect accuracy (R² = 1.0) in forecasting monthly sentiment scores

## Top 3 Positive & Negative Employees (Example Month: 2011-11)
**Top 3 Positive:**
1. patti.thompson@enron.com (Score: +7)
2. don.baughman@enron.com (Score: +2)
3. eric.bass@enron.com (Score: +1)

**Top 3 Negative:**
1. bobette.riner@ipgdirect.com (Score: -8)
2. kayne.coulter@enron.com (Score: -7)
3. john.arnold@enron.com (Score: -4)

## Flight Risk Employees
All 10 employees were identified as flight risks:
- bobette.riner@ipgdirect.com
- don.baughman@enron.com
- eric.bass@enron.com
- john.arnold@enron.com
- johnny.palmer@enron.com
- kayne.coulter@enron.com
- lydia.delgado@enron.com
- patti.thompson@enron.com
- rhonda.denton@enron.com
- sally.beck@enron.com

## Key Insights & Recommendations
- High negative sentiment (57.69%) suggests potential stress or dissatisfaction
- All employees show flight risk indicators — urgent retention review recommended
- Message volume and negative message count strongly predict sentiment scores
- Consider targeted engagement initiatives and follow-up surveys

See `visualizations/` folder for charts and `processed/labeled_data.csv` for full labeled dataset.