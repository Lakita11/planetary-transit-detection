# 🌌 Kepler Exoplanet Transit Detection — ML Classifier Comparison

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![SMOTE](https://img.shields.io/badge/SMOTE-imbalanced--learn-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A complete data science pipeline for classifying exoplanet transit candidates from the NASA Kepler mission — with a focus on solving the **class imbalance problem** that most existing studies ignore.

---

## 📖 About the Project

NASA's Kepler Space Telescope monitored over 150,000 stars from 2009–2018 and produced the **Kepler Objects of Interest (KOI)** dataset — thousands of potential exoplanet transit candidates waiting to be confirmed or rejected.

The catch? The dataset is heavily imbalanced:

| Class | Count | Percentage |
|---|---|---|
| FALSE POSITIVE | ~802 | 44.8% |
| CONFIRMED (planet) | ~631 | 35.2% |
| CANDIDATE | ~358 | 20.0% |

Most studies just report overall accuracy — which is **completely misleading** here. A model that always predicts "not a planet" would score 64% accuracy without finding a single real planet.

This project:
- Cleans a messy, real-world version of the KOI dataset (missing values, wrong labels, impossible values, unit errors)
- Compares **5 classical ML classifiers** using proper imbalance-aware metrics (F1, Precision, Recall, AUC-ROC)
- Applies **SMOTE** to fix class imbalance and measures its effect
- Uses **SHAP** to explain which features drive the predictions

---

## 📁 Project Structure

```
kepler-transit-detection/
│
├── Kepler_Notebook_Clean.ipynb      ← Main notebook (run this)
├── kepler_transit_detection_messy.csv  ← Raw messy dataset
├── kepler_cleaned.csv               ← Auto-generated after cleaning
├── README.md                        ← This file
│
├── figures/                         ← Auto-saved when notebook runs
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── period_vs_radius.png
│   ├── f1_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_importance.png
│   └── shap_beeswarm.png
│
└── Research_Paper_Humanized.docx    ← Research paper (IEEE format)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn 1.4 |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Gradient Boosting | XGBoost 2.0 |
| Interpretability | SHAP |
| Notebook | Jupyter |

---

## ⚙️ How to Run

### Option A — Google Colab (recommended, no setup needed)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `Kepler_Notebook_Clean.ipynb`
3. Upload `kepler_transit_detection_messy.csv` to the Colab file browser (left sidebar)
4. Click **Runtime → Run all**

### Option B — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/kepler-transit-detection.git
cd kepler-transit-detection
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook Kepler_Notebook_Clean.ipynb
```

**4. Run all cells top to bottom** using Shift + Enter

---

## 🧹 Data Cleaning Pipeline

The raw dataset had 8 types of data quality issues — all fixed in the notebook:

| Step | Problem | Fix |
|---|---|---|
| 1 | 15+ spelling variants of class labels | Mapped all to 3 canonical values |
| 2 | String garbage in numeric cols ("N/A", "--", "?") | `pd.to_numeric(errors='coerce')` |
| 3 | Physically impossible values (negative periods, score > 1) | Boolean filter and drop |
| 4 | Transit duration in wrong units (minutes vs hours) | Divide by 60 where value > 15 |
| 5 | Duplicate Kepler IDs | `drop_duplicates(subset=['kepid'])` |
| 6 | Flag columns with invalid values (2, -1, "Y", "N") | Coerce to 0/1 binary |
| 7 | Missing values | Median imputation (drop if > 40% missing) |
| 8 | Feature scale differences | StandardScaler (fit on train only) |

---

## 📊 Results

### Classifier Comparison (F1-Score for Confirmed Planet class)

| Classifier | F1 (no SMOTE) | F1 (with SMOTE) | AUC (SMOTE) |
|---|---|---|---|
| Logistic Regression | 0.946 | **0.957** | 0.992 |
| Decision Tree | 0.961 | 0.926 | 0.944 |
| Random Forest | 0.957 | 0.957 | 0.995 |
| SVM | 0.964 | 0.940 | 0.990 |
| XGBoost | 0.952 | 0.948 | 0.995 |

**Best model:** Logistic Regression with SMOTE — F1 = 0.957, AUC = 0.992

### Key Findings

- **Class imbalance matters** — without SMOTE, XGBoost Recall drops to 0.944 despite a high AUC
- **SMOTE helps linear models most** — LR Recall stays at 0.968 while Precision improves
- **SMOTE hurts Decision Trees** — F1 drops from 0.961 to 0.926 due to synthetic sample noise at boundaries
- **LR beats ensemble methods** — because `koi_score` alone creates near-linear separability (50.9% Gini importance)

### Top Features (SHAP Importance)

| Rank | Feature | Description | Importance |
|---|---|---|---|
| 1 | `koi_score` | Kepler pipeline disposition score | 50.9% |
| 2 | `koi_snr` | Signal-to-noise ratio | 12.8% |
| 3 | `koi_fpflag_nt` | Non-transit-like false positive flag | 10.2% |
| 4 | `koi_depth` | Transit depth (ppm) | 5.8% |
| 5 | `koi_prad` | Planet radius (Earth radii) | 5.2% |

---

## 📉 Visualisations

The notebook auto-generates and saves 10 figures:

| Figure | What it shows |
|---|---|
| Class Distribution | The imbalance problem visualised (bar + pie) |
| Feature Distributions | KDE curves per class for 6 key features |
| Correlation Heatmap | Feature-to-feature Pearson correlations |
| Boxplots | Spread and outliers by class |
| Period vs Radius | Scatter showing class separation |
| F1 Comparison | With vs without SMOTE bar chart |
| ROC Curves | AUC for all 5 classifiers |
| Confusion Matrices | TP/TN/FP/FN for all 5 classifiers |
| SHAP Importance | Which features matter most |
| SHAP Beeswarm | Direction and magnitude of each feature's effect |

---

## 🔬 Research Paper

This project is accompanied by a full research paper in **IEEE conference format**:

> *"Comparative Analysis of Machine Learning Classifiers for Imbalanced Exoplanet Transit Detection Using the Kepler KOI Dataset"*

The paper covers: Introduction, Related Work, Dataset & Preprocessing, Methodology, Results, Discussion, Conclusion, and 16 references from published astronomy and ML literature.

---

## 📚 References

1. Borucki et al. (2010) — Kepler Planet-Detection Mission, *Science*
2. He & Garcia (2009) — Learning from Imbalanced Data, *IEEE TKDE*
3. Shallue & Vanderburg (2018) — AstroNet, *Astronomical Journal*
4. Valizadegan et al. (2022) — ExoMiner, *Astrophysical Journal*
5. Kovács et al. (2002) — BLS Algorithm, *A&A*
6. Hippke & Heller (2019) — TLS Algorithm, *A&A*
7. Chawla et al. (2002) — SMOTE, *JAIR*
8. Schanche et al. (2019) — ML for WASP Survey, *MNRAS*
9. Raj & Kos (2025) — CNN for Kepler Data, *Scientific Reports*
10. Akeson et al. (2013) — NASA Exoplanet Archive, *PASP*

---

## 👤 Author

**[Your Full Name]**
2nd Year B.Tech — Computer Science & Engineering
[Your Institution Name]
[your.email@institution.ac.in]

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share with attribution.

---

*Dataset source: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Kepler KOI Cumulative Table (Q1–Q17, DR25)*
