# Employee Sentiment Analysis

## Summary
This project analyzes internal employee emails to assess sentiment trends, rank employees by communication tone, detect potential flight risks, and predict sentiment scores using machine learning.

---

## Top Employees

### **Top 5 Positive Employees**
- `patti.thompson@enron.com` —> **+30**
- `sally.beck@enron.com` —> **+24**
- `john.arnold@enron.com` —> **+20**
- `eric.bass@enron.com` —> **+20**
- `bobette.riner@ipgdirect.com` —> **+19**

### **Top 3 Negative Employees**
- `rhonda.denton@enron.com` —> **-6**
- `patti.thompson@enron.com` —> **-4**
- `don.baughman@enron.com` —> **-4**
- `john.arnold@enron.com` —> **-3**
- `sally.beck@enron.com` —> **-3**


---

## Flight Risk Employees
Employees flagged as **flight risks** (≥4 negative messages in any rolling 30-day window):
- `bobette.riner@ipgdirect.com`
- `don.baughman@enron.com`
- `eric.bass@enron.com`
- `john.arnold@enron.com`
- `johnny.palmer@enron.com`

---

## Key Insights
1. **Positive sentiment** dominates overall communication, but certain months show spikes in negative sentiment, indicating possible morale issues.
2. A **small group of employees** sends the majority of emails, with consistent communication styles (either highly positive or consistently negative).
3. Flight risk employees often overlap with the monthly **Top Negative Employees**, reinforcing the accuracy of the detection logic.
4. **Monthly message count** is the most influential factor in predicting sentiment scores, followed by **average word count**.

---

## Recommendations
- **Engage high scorers** in employee initiatives—they are potential morale multipliers.
- **Monitor consistently negative employees** and offer targeted support or interventions.
- Track sentiment trends **monthly** to spot early warning signs.
- Incorporate **department-level** or **project-level context** to refine analysis and predictions.
- Expand the model with additional features such as **email response time** or **network centrality** for richer insights.