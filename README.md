<div align="center">

<img src="https://img.shields.io/badge/IBM%20SkillsBuild-Data%20Analytics%20%26%20AI-054ADA?style=for-the-badge&logo=ibm&logoColor=white"/>
<img src="https://img.shields.io/badge/BharatCares%20×%20AICTE-Internship-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Complete%20✓-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/GitHub%20Pages-Live%20Site-0f3460?style=for-the-badge&logo=github&logoColor=white"/>

# 🎓 AI Impact on Students
### Data Analytics & Machine Learning Project

*Analyzing how Generative AI tools, study habits, AI dependency, and prompt skills*  
*affect student academic outcomes, GPA changes, and burnout risk*

---

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4CAAD4?style=flat-square)](https://seaborn.pydata.org/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-8.0+-7A29FF?style=flat-square)](https://ipywidgets.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 👤 Author

| | |
|---|---|
| **Name** | Dharamveer Sharma |
| **Program** | IBM SkillsBuild Data Analytics with AI Academic Internship |
| **Organized by** | BharatCares in association with AICTE |
| **Repository** | `IBM_Skillbuild_project` |
| **Dataset** | AI Impact on Students — Kaggle (50,000 records, 16 features) |

---

## 📌 Project Overview

This project investigates the **multi-dimensional impact of Generative AI** on student academic performance using a structured dataset of **50,000 students** across five major categories and five academic year levels.

```
Key Research Questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 📈  Does more GenAI usage improve or hurt GPA?
 🤖  Does AI dependency reduce skill retention?
 🧠  Do better prompt skills lower burnout risk?
 🏫  Which institutional policy produces best outcomes?
 🔥  Can we predict student burnout risk with ML?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 Dataset at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  AI Impact on Students Dataset  ·  Source: Kaggle                  │
├──────────────────┬──────────────────────────────────────────────────┤
│  Total Records   │  50,000 students                                 │
│  Features        │  16 columns (expanded to 20 after engineering)   │
│  Missing Values  │  None — clean and complete                       │
│  Target Classes  │  Low (32.7%)  ·  Medium (42.3%)  ·  High (25.0%)│
├──────────────────┴──────────────────────────────────────────────────┤
│  Feature Groups                                                      │
│  ├── 📚 Academic    Pre/Post GPA, Traditional Study Hours           │
│  ├── 🤖 AI Usage    Weekly GenAI Hours, Tool Diversity, Use Case    │
│  ├── 🧠 Cognition   Prompt Skill, AI Dependency, Skill Retention    │
│  ├── 🏫 Contextual  Major, Year, Institutional Policy               │
│  └── 🎯 Target      Burnout Risk Level (Low / Medium / High)        │
└─────────────────────────────────────────────────────────────────────┘
```

| Feature | Type | Range / Values |
|---|---|---|
| `Major_Category` | Categorical | STEM · Humanities · Business · Arts · Medical |
| `Year_of_Study` | Ordinal | Freshman → Sophomore → Junior → Senior → Graduate |
| `Pre_Semester_GPA` / `Post_Semester_GPA` | Float | 0.0 – 4.0 |
| `Weekly_GenAI_Hours` | Float | 0 – 40 hrs/week |
| `Perceived_AI_Dependency` | Integer | 1 – 10 scale |
| `Prompt_Engineering_Skill` | Ordinal | Beginner · Intermediate · Advanced |
| `Skill_Retention_Score` | Float | 0 – 100 |
| `Institutional_Policy` | Categorical | Strict_Ban · Allowed_With_Citation · Actively_Encouraged |
| `Burnout_Risk_Level` | **Target** | **Low · Medium · High** |

---

## 🏗️ Project Pipeline

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐
│  Step 0 │───▶│  Step 1  │───▶│  Steps 2–4   │───▶│   Step 5   │───▶│  Step 6  │
│ Install │    │ Imports  │    │ Load · Clean │    │  Engineer  │    │   EDA    │
│  deps   │    │ & config │    │  & validate  │    │  features  │    │  5 plots │
└─────────┘    └──────────┘    └──────────────┘    └────────────┘    └────┬─────┘
                                                                           │
               ┌──────────┐    ┌──────────────┐    ┌────────────┐         │
               │  Step 9  │◀───│   Step 8     │◀───│   Step 7   │◀────────┘
               │Conclusion│    │  Insights    │    │  ML Train  │
               │ & Ethics │    │  & summary   │    │  RF + LR   │
               └──────────┘    └──────────────┘    └────────────┘
```

**5 Engineered Features** added in Step 5:

| Feature | Formula | Purpose |
|---|---|---|
| `GPA_Change` | Post GPA − Pre GPA | Academic trajectory |
| `AI_Study_Ratio` | GenAI hrs ÷ (GenAI + Study hrs) | AI integration depth |
| `Skill_Efficiency` | Skill Retention ÷ (GenAI hrs + 1) | Quality per AI hour |
| `High_AI_Dependency` | 1 if Dependency ≥ 5 else 0 | Binary risk flag |
| `Prompt_Skill_Numeric` | Beginner=1, Intermediate=2, Advanced=3 | Ordinal for correlation |

---

## 📈 Visualizations

> All plots are automatically generated and saved as high-resolution PNGs when the notebook is run.

| # | File | Description |
|---|---|---|
| 1 | `plot_gpa_change.png` | GPA change histogram + KDE, and box-plot stratified by burnout risk |
| 2 | `plot_ai_dependency_skill.png` | Scatter of AI dependency vs skill retention + avg retention bar by prompt skill |
| 3 | `plot_genai_gpa_major.png` | GenAI hours vs GPA regression scatter + burnout count bar by major |
| 4 | `plot_correlation_heatmap.png` | Full lower-triangle Pearson correlation heatmap (12 numeric features) |
| 5 | `plot_policy_analysis.png` | AI-to-study ratio violin by policy + avg exam anxiety bar grid |
| 6 | `plot_confusion_matrices.png` | Side-by-side confusion matrices for RF and LR models |
| 7 | `plot_feature_importance.png` | Top 15 RF feature importances, colour-coded by tier |

---

## 🛠️ Technologies Used

```
Core Data Stack
────────────────────────────────────────────────────────────
  🐍 Python 3.13+        Runtime
  🐼 Pandas 2.0+         Data loading, cleaning, feature engineering
  🔢 NumPy 1.24+         Numerical operations & array math

Visualisation
────────────────────────────────────────────────────────────
  📊 Matplotlib 3.7+     Static EDA plots (5 figures × 2 subplots)
  🎨 Seaborn 0.12+       Statistical charts (heatmap, violin, box, KDE)
  📉 Plotly 5.18+        Interactive scatter chart with hover profiles

Machine Learning
────────────────────────────────────────────────────────────
  🌲 RandomForestClassifier   Primary model  (300 trees, balanced)
  📐 LogisticRegression       Baseline model (lbfgs, multinomial)
  ⚙️  scikit-learn 1.5+        Preprocessing, metrics, CV

Interactivity & UI
────────────────────────────────────────────────────────────
  🎛️  ipywidgets 8.0+     Dataset Explorer + Burnout Predictor widgets
  🖥️  IPython.display     Styled HTML banners, insight cards, tables
  📓 Jupyter 7.0+        Notebook environment
```

---

## 🎛️ Interactive Features

The notebook contains **3 interactive elements** — all update in real time:

```
┌──────────────────────────────────────────────────────────────────┐
│  📍 Step 6.6 — Plotly Scatter Chart                              │
│  Hover any student point to reveal their full profile:           │
│  Major, Dependency score, Anxiety level, Study hours             │
├──────────────────────────────────────────────────────────────────┤
│  🔎 Step 6.7 — Dataset Explorer  (ipywidgets)                    │
│  3 dropdowns: Major · Year · Metric                              │
│  → KDE distribution + Box plot update instantly                  │
├──────────────────────────────────────────────────────────────────┤
│  🤖 Step 7.8 — Burnout Risk Predictor  (ipywidgets)              │
│  6 sliders + 6 dropdowns → live RF model prediction              │
│  → Colour-coded badge  +  3-class probability bar chart          │
└──────────────────────────────────────────────────────────────────┘
```

> ⚠️ **Requirements:** Widgets need `ipywidgets` (auto-installed by Step 0) and a **live Jupyter kernel**. They will not render on GitHub preview or nbviewer.

---

## ⚙️ Setup & Run Instructions

### Step 1 — Clone

```bash
git clone https://github.com/<your-username>/IBM_Skillbuild_project.git
cd IBM_Skillbuild_project
```

### Step 2 — Virtual Environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Launch & Run

```bash
jupyter notebook
# Open: DharamveerSharma_AI_Impact_Students.ipynb
# Then: Kernel → Restart & Run All
```

> 📁 Make sure `ai_student_impact_dataset (1).csv` is in the **same folder** as the notebook.

---

## 🔍 Key Analytical Findings

```
 Finding                          Insight
 ─────────────────────────────────────────────────────────────────────
 📉  High burnout → lower GPA     High-risk students show lowest avg
                                  GPA change (+0.196 vs +0.200 Low)
 
 🤖  AI dependency hurts          Correlation = −0.084 between
     skill retention              dependency and retention score
 
 ✏️   Prompt skill is protective   Advanced users have lower High-
                                  burnout rate than Beginners
 
 ⏰  Traditional study counts      Remains a top-6 RF feature;
                                  AI supplements, not replaces
 
 🏫  Policy shapes behaviour      Actively-Encouraged → highest AI
                                  ratio with widest variance
 
 🌲  Random Forest outperforms    RF: 51.9% vs LR: 51.1%
                                  Both beat 42.3% majority baseline
 ─────────────────────────────────────────────────────────────────────
```

---

## 🤖 ML Model Results

```
┌────────────────────────┬──────────────┬─────────────────────┬───────────┐
│  Model                 │ Test Accuracy│  CV Accuracy (5-fold)│  vs Base  │
├────────────────────────┼──────────────┼─────────────────────┼───────────┤
│  Random Forest ✓ Best  │    51.87%    │     ~51% ± 0.3%     │  + 9.6 pp │
│  Logistic Regression   │    51.12%    │     ~51% ± 0.4%     │  + 8.8 pp │
│  Majority-class Base   │    42.29%    │        42.29%       │    —      │
└────────────────────────┴──────────────┴─────────────────────┴───────────┘

  ⓘ  Both models beat the majority-class baseline, confirming real signal
     in the intentionally-noisy synthetic dataset (3 closely-spaced classes).
```

**Top 6 Predictors of Burnout Risk** (by RF feature importance):

```
  1. 🕐 Weekly GenAI Hours           ████████████████████  Highest
  2. ⚖️  AI-to-Study Ratio            ████████████████
  3. 🤖 Perceived AI Dependency      █████████████
  4. 📐 Skill Efficiency             ████████████
  5. 🚩 High AI Dependency (flag)    ██████████
  6. 😰 Exam Anxiety Level           ████████
```

---

## 📋 Ethical Recommendations

| # | Recommendation | Rationale |
|---|---|---|
| 1 | 🎓 **Embed AI Literacy in curricula** | Prompt engineering should be a formal competency |
| 2 | ⚖️ **Adopt citation-based AI policies** | Blanket bans increase anxiety without improving outcomes |
| 3 | 🚨 **Deploy early-warning systems** | ML burnout predictors can flag at-risk students mid-semester |
| 4 | 📚 **Preserve traditional study time** | Top predictor of GPA growth; AI must supplement, not replace |
| 5 | 🔒 **Govern data use transparently** | Any predictive deployment requires student consent & privacy controls |

---

## 📂 Repository Structure

```
IBM_Skillbuild_project/
│
├── 📓 DharamveerSharma_AI_Impact_Students.ipynb  ← Main notebook (56 cells)
├── 📄 DharamveerSharma_ProjectReport.docx        ← Full project report
├── 📊 ai_student_impact_dataset (1).csv          ← Raw dataset (50k rows)
├── 📦 requirements.txt                           ← Python dependencies
├── 📖 README.md                                  ← This file
│
└── 📁 Generated plots (created on notebook run)
    ├── plot_gpa_change.png
    ├── plot_ai_dependency_skill.png
    ├── plot_genai_gpa_major.png
    ├── plot_correlation_heatmap.png
    ├── plot_policy_analysis.png
    ├── plot_confusion_matrices.png
    └── plot_feature_importance.png
```

---

## 🌐 GitHub Pages — Live Website

This project ships with a complete **GitHub Pages website** in the `docs/` folder.

### Repo Structure for Pages
```
IBM_Skillbuild_project/
├── docs/
│   ├── index.html               ← Full project website (self-contained)
│   ├── .nojekyll                ← Bypasses Jekyll; serves raw HTML
│   ├── _config.yml              ← Site title & description
│   ├── DharamveerSharma_AI_Impact_Students.ipynb  ← Downloadable from site
│   ├── DharamveerSharma_ProjectReport.docx        ← Downloadable from site
│   └── assets/
│       ├── plot_gpa_change.png
│       ├── plot_ai_dependency_skill.png
│       ├── plot_genai_gpa_major.png
│       ├── plot_correlation_heatmap.png
│       ├── plot_policy_analysis.png
│       ├── plot_confusion_matrices.png
│       └── plot_feature_importance.png
├── .github/workflows/pages.yml  ← Auto CI/CD deploy on push
└── ...
```

### Deploy in 3 Steps

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit: AI Impact on Students project"
git branch -M main
git remote add origin https://github.com/<your-username>/IBM_Skillbuild_project.git
git push -u origin main
```

**Step 2 — Enable GitHub Pages**
```
GitHub → Your Repo → Settings → Pages
→ Source: "Deploy from a branch"
→ Branch: main  /  Folder: /docs
→ Click Save
```

**Step 3 — Your site goes live at:**
```
https://<your-username>.github.io/IBM_Skillbuild_project/
```

> ✅ After the first push, every subsequent `git push` **auto-deploys** via the included
> `.github/workflows/pages.yml` GitHub Actions workflow — no manual steps needed.

### What the website includes
| Feature | Details |
|---|---|
| 🧭 Sticky navigation | Scrolls to each section smoothly |
| 📊 All 7 plot images | Click any plot to open full-size lightbox |
| 📈 Animated progress bars | Model accuracy bars animate on scroll |
| 🎨 Scroll fade-in | Cards and sections animate in as you scroll |
| 📱 Fully responsive | Works on desktop, tablet, and mobile |
| 💾 Download buttons | Notebook + Report downloadable directly from site |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

*Submitted as part of the IBM SkillsBuild Data Analytics with AI Academic Internship*
*Organized by BharatCares × AICTE | Author: Dharamveer Sharma*

</div>
