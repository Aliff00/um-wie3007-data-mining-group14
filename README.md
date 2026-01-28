# WIE3007 Group Project (2025/2026 Sem 1)

**Course:** WIE3007 – Data Mining  
**Faculty:** Faculty of Computer Science and Information Technology, Universiti Malaya  
**Project:** AI‑enhanced analytics in a financial/business domain  
**Group Size:** Maximum 5 members  
**Due:** Week 14

## Project Overview
This project applies data mining workflows with Generative AI (GenAI), Large Language Models (LLMs), and Small Language Models (SLMs) to a simulated financial/business dataset. We perform dataset simulation, feature engineering, predictive modelling, and model interpretation, with clear documentation of AI usage and GitHub contributions.

## Objectives
- Simulate or generate a realistic financial/business dataset (≥ 1000 records).
- Use LLMs/SLMs to extract features (e.g., sentiment, risk category, customer segment).
- Build a Decision Tree baseline and at least four additional models.
- Evaluate models with appropriate metrics and interpret results with AI support.
- Produce a professional report and presentation, with AI usage disclosure.

## Repository Structure
- `data/` — Raw and processed datasets.
- `notebooks/` — Jupyter notebooks for simulation, EDA, feature engineering, and modelling.
- `outputs/` — Model results, figures, and exported artifacts.
- `report/` — Final report drafts and assets.
- `bin/` — Utility scripts (data prep, training, evaluation).
- `dataset_generation_prompt.txt` — Prompt used for AI‑assisted dataset simulation.

## Assignment Requirements (Mapped)
### 1) Dataset Simulation & Feature Engineering (4 marks)
- Generate ≥ 1000 financial/business records.
- Use GenAI to create realistic numeric + text fields.
- Extract AI features (e.g., sentiment, risk, segmentation).

### 2) Predictive Model Development (5 marks)
- **Baseline:** Decision Tree (mandatory)
- **Additional models (choose ≥ 4):**
  - Random Forest
  - XGBoost / Gradient Boosting
  - Logistic Regression
  - Neural Network (MLP)
  - k‑Nearest Neighbors (k‑NN)
  - Naive Bayes

Each member is responsible for at least one model with evidence in commits and report.

### 3) Model Evaluation & Interpretation (4 marks)
- Evaluate with suitable metrics (Accuracy, F1, ROC‑AUC, RMSE, etc.).
- Use LLMs to summarize findings, interpret feature importance, and provide insights.

### 4) Final AI‑Assisted Report (2 marks)
- 5–7 pages covering: objectives, dataset, EDA, features, modelling, results, insights,
  AI usage disclosure, and GitHub contribution summary.

## GitHub Contribution Policy (Compulsory)
To meet the course requirement, **each member must**:
- Make at least 6 commits between Weeks 7–13.
- Work on their **own branch**.
- Write meaningful commit messages.
- Contribute to at least one notebook section and one modelling task.
- Review or comment on at least one pull request.

Failure to demonstrate individual contributions may result in deductions.

## Deliverables
1. GitHub repository
2. Jupyter notebook(s)
3. Final report (PDF/DOCX)
4. Recorded presentation

## How to Run (Local)
> Update this section once the final environment and entry points are fixed.

- Create a Python environment and install dependencies.
- Run notebooks in `notebooks/` for data simulation, EDA, and modelling.
- Export figures and metrics to `outputs/`.

## Team & Roles
## Contribution Summary
| Name | Contribution |
| --- | --- |
| Muhamad Aliff Najmi | Dataset simulation, Decision Tree, model performance comparison, LLM model interpretation |
| Zikriaiman | Part of feature engineering, Naive Bayes |
| Darwis | Part of feature engineering, XGBoost |
| Nurhazimah Diyana Sofea | Part of data cleaning, k-NN, business insights |
| Nurul Aisyah | Part of data cleaning, EDA, Logistic Regression |

## AI Usage Disclosure
We document all AI tools, prompts, and outputs used for dataset simulation, feature extraction,
model interpretation, and report writing. Relevant prompts are stored in `dataset_generation_prompt.txt`
and in notebook sections where applicable.

---
If you are a contributor, please follow the GitHub policy above and keep commits well‑scoped and descriptive.
