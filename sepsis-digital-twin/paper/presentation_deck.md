# Presentation Slide Deck & Presentation Script

This presentation deck is designed for a 5-7 minute conference presentation. It is formatted in standard Markdown with horizontal dividers (`---`) separating slides, and includes speaker notes for each slide.

---

# Slide 1: Title
## Leakage-Aware and Explainable Cross-Hospital Sepsis Early-Warning
### Using Dynamic Patient-State Modelling
**Authors**: Lead Implementation Agent, Google DeepMind Advanced Agentic Coding Team  
**Date**: August 27, 2026

---
### Speaker Notes:
"Good morning, everyone. Today, I am excited to present our research on a leakage-aware and explainable cross-hospital sepsis early-warning system. Our work leverages dynamic patient-state modelling to build highly transportable machine learning classifiers that predict sepsis onset 6 hours in advance while strictly preventing temporal lookahead bias during preprocessing."

---

# Slide 2: Clinical Motivation
* **Sepsis**: A life-threatening organ dysfunction caused by a dysregulated host response to infection.
* **Clinical Window**: Early detection is crucial; every hour of delayed treatment increases mortality.
* **Early Warning**: Our target is to predict sepsis onset **6 hours in advance** to guide timely clinical intervention.
* **The Challenges**:
  1. **Data Leakage**: Preprocessing steps (e.g., forward-filling labs) often introduce future lookahead bias.
  2. **Generalizability**: Models trained on one hospital frequently fail when transported to another hospital.

---
### Speaker Notes:
"Sepsis is a leading cause of death in intensive care units worldwide. While machine learning holds great promise for early warning, existing models face two major barriers: first, data leakage during preprocessing where future information is inadvertently used, creating artificially high performance in retrospective studies; and second, poor generalizability when transported across different hospital environments. We address both issues in this study."

---

# Slide 3: Leakage-Aware Preprocessing
* **Causal Preprocessing**:
  * Sort all measurements strictly by ICU length of stay (`ICULOS`).
  * Restrict forward-filling to a maximum of **24 hours**.
  * Never use future observations to fill values at hour $t$.
* **Missingness Attributes**:
  * Missingness indicators (`col_missing`) track observation patterns.
  * Recency indicators (`col_recency`) track hours elapsed since the last measurement.
* **Zero Cross-Hospital Leakage**:
  * Imputation statistics (medians) are fitted **only** on Hospital A training data.
  * No information from the Hospital B holdout dataset is ever seen during development.

---
### Speaker Notes:
"To prevent lookahead bias, we implemented a causal preprocessor. Measurements are sorted strictly chronologically by length of stay, and we limit forward-filling of clinical variables to 24 hours. Crucially, we build explicit missingness and recency indicators that act as proxies for clinical monitoring frequency. Finally, all imputation parameters are fitted strictly on the Hospital A training cohort, ensuring zero information leak from Hospital B."

---

# Slide 4: Dynamic Patient-State Modelling
* **Digital-Twin-Inspired Dynamic Trajectories**:
  * Extract features using backward-looking sliding windows of 3 and 6 hours.
* **Feature Construction**:
  * **Current Value**: $x_t$
  * **Rolling Mean**: $\mu_{{3h}}, \mu_{{6h}}$
  * **Rolling Variability**: $\sigma_{{3h}}, \sigma_{{6h}}$
  * **Trend**: Least-squares linear regression slope.
  * **Change**: Delta values ($x_t - x_{{t-k}}$).
* **Causal Focus**: No centered or future-looking windows are used.

---
### Speaker Notes:
"Instead of static measurements, we construct dynamic patient trajectories inspired by digital twins. We calculate rolling means, standard deviations, delta changes, and least-squares regression slopes over the past 3 and 6 hours. Because these calculations look exclusively backward, they capture vital sign and lab trajectories causally, with zero lookahead leakage."

---

# Slide 5: Models & Cross-Hospital Evaluation
* **Models**:
  1. **Baseline**: Logistic Regression (class-weighted).
  2. **Primary Model**: XGBoost (class imbalance adjusted via `scale_pos_weight`, hyperparameter tuned).
  3. **Sequence Model**: LSTM (Masked recurrent neural network trained on sequence trajectories).
* **Dataset Splitting**:
  * **Development (Hospital A)**: 20,336 patients (80% Train, 20% Val).
  * **Held-out Test (Hospital B)**: 20,000 patients (entirely held-out, unlocked only at final freeze).
  * **No Patient Overlap**: Split is patient-wise, preserving patient integrity.

---
### Speaker Notes:
"We developed three models: a baseline Logistic Regression, a primary XGBoost classifier, and an LSTM sequence model built in Keras to process temporal patient trajectories. We split our data patient-wise: Hospital A with over 20,000 patients was partitioned 80/20 for training and validation. Hospital B with 20,000 patients was completely held out, serving as a transportability test set representing a different hospital system."

---

# Slide 6: Main Performance Results
* **Transportability & Evaluation**:
  * XGBoost demonstrates strong cross-hospital transportability.
  * The table below compares validation performance to holdout test performance:

| Model / Metric | Hospital A Validation | Hospital B Test | Absolute Change | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost AUROC** | 0.8264 | 0.7753 (0.7622, 0.7892) | -0.0511 | -6.18% |
| **XGBoost Utility** | 0.4589 | 0.2628 (0.2444, 0.2898) | -0.1961 | -42.73% |
| **LR AUROC** | 0.8011 | 0.6686 (0.6514, 0.6837) | -0.1325 | -16.54% |
| **LR Utility** | 0.4128 | 0.1041 (0.0748, 0.1306) | -0.3087 | -74.78% |
| **LSTM AUROC** | 0.7549 | 0.6916 (0.6786, 0.7036) | -0.0633 | -8.38% |
| **LSTM Utility** | 0.3240 | 0.0126 (-0.0166, 0.0440) | -0.3114 | -96.11% |

*(Note: Complete metrics are compiled in the manuscript)*

---
### Speaker Notes:
"Looking at the performance results, we observe that the XGBoost model generalizes remarkably well. Its AUROC drops by only 6.18% relative when transported to the held-out Hospital B, achieving a test AUROC of 0.7753. In contrast, the Logistic Regression baseline suffers a much steeper drop of 16.54% relative, falling to an AUROC of 0.6686. This highlights the value of non-linear models combined with dynamic features for cross-hospital transportability."

---

# Slide 7: Warning Time and Alarm Burden
* **Warning Lead Time**:
  * XGBoost achieved a median warning lead time of **32.0 hours** before clinical sepsis onset.
  * Enables proactive intervention rather than reactive treatment.
* **False Alarm Burden**:
  * False alert rate of **9.06 false alerts per 100 patient-hours**.
  * Highly manageable alert burden for clinical staff, reducing alarm fatigue.

---
### Speaker Notes:
"A key metric for clinical utility is warning lead time. Our primary model achieved a median warning lead time of 32 hours before clinical onset of sepsis, providing a wide window for clinicians to initiate therapy or order blood cultures. Furthermore, the alarm burden is highly manageable, producing only 9.06 false alerts per 100 patient-hours, which helps mitigate the critical issue of clinical alarm fatigue."

---

# Slide 8: Explainability (SHAP & Case Studies)
* **Global Attributions**:
  * SHAP analysis reveals that temperature slope, heart rate rolling mean, and respiration rate are key predictors.
  * Measurement recency (time since lactate and pH were last checked) are highly predictive, highlighting clinical workflows as proxy signals.
* **Explainability vs. Causality**:
  * SHAP scores represent statistical associations, **not** physiological causality.
* **Per-Patient Timeline Case Studies**:
  * Timeline plots track risk probabilities alongside physiological variables (HR, Temp) for validation.

---
### Speaker Notes:
"For explainability, we computed SHAP attributions. Temperature slope, heart rate rolling mean, and respiration rate were key predictors. Intriguingly, the recency of lactate and pH tests were also highly predictive. This shows that the model is exploiting clinical workflows—when a clinician orders a lactate test, it is a strong proxy that the patient is deteriorating. We emphasize that SHAP indicates statistical association, not physiological causality."

---

# Slide 9: Benchmark against Moore et al. (2020)
* **Moore et al. (n=29)**:
  * Hybrid model (Bayesian networks, agent-based simulation) predicting specific treatment responses (meds, fluids, ventilation) during the first 24h of sepsis.
* **Methodological Comparison**:
  * **Our Work (n=40,336)**: Large multicenter cohort, focused on early warning/onset prediction.
  * **Causal vs. Statistical**: Moore et al. focus on physiological mechanism/treatment response; our work focuses on predictive risk modeling.
  * **Conclusion**: Direct numerical comparison is scientifically inappropriate due to completely different tasks, but they serve as complementary paradigms.

---
### Speaker Notes:
"We benchmarked our work qualitatively against Moore et al. (2020), who developed a digital-twin model for 29 patients to predict specific treatment responses. While both projects use digital-twin concepts, they are structurally different: Moore et al. model causal, mechanistic responses to interventions in a small cohort, whereas we perform large-scale statistical sepsis early-warning. A direct quantitative comparison of accuracy is scientifically inappropriate, but they represent complementary and valuable paradigms in ICU modeling."

---

# Slide 10: Limitations & Future Work
* **Limitations**:
  * Retrospective study (proof of concept only).
  * **Not** clinically validated for bedside deployment.
  * Exploits clinical measurement frequency (which may vary by hospital).
* **Future Work**:
  * Prospective silent trial validation.
  * Integration of real-time clinical note embeddings.
  * Dynamic counterfactual modeling to predict response to specific treatments (merging our paradigm with Moore et al.).

---
### Speaker Notes:
"In conclusion, our study is a retrospective proof of concept and is not ready for bedside deployment. Future work should focus on prospective silent trials to validate transportability in real time, and merging our early-warning model with mechanistic models like Moore's to predict not just *when* a patient will develop sepsis, but *how* they will respond to specific therapeutic choices. Thank you, and I am happy to take any questions."
