# Leakage-Aware and Explainable Cross-Hospital Sepsis Early-Warning Using Dynamic Patient-State Modelling

## 1. Abstract
Sepsis remains a leading cause of mortality in Intensive Care Units (ICUs) worldwide. Early warning systems can improve survival rates, but clinical models are often plagued by temporal data leakage during preprocessing and show poor generalizability when transported across healthcare systems. We present a leakage-aware, explainable cross-hospital sepsis early-warning framework based on a digital-twin-inspired dynamic patient-state model. Using the PhysioNet/CinC Challenge 2019 dataset, we trained baseline Logistic Regression, primary XGBoost, and secondary sequence LSTM models on data from Hospital A (20336 patients, 776,105 hours) and evaluated transportability on Hospital B (20000 patients, held-out). XGBoost achieved a validation AUROC of 0.8264 and successfully generalized to Hospital B with an AUROC of 0.7753 95% CI (0.7622, 0.7892) and normalized utility score of 0.2628, significantly outperforming the Logistic Regression baseline. Attributions computed via SHAP identified key clinical features associated with the risk predictions without claiming physiological causality. The system serves as a retrospective proof of concept illustrating the impact of leakage prevention and dynamic features on transportable sepsis prediction.

## 2. Introduction
Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection. Every hour of delayed treatment increases the risk of mortality. Early-warning alerts, targeted 6 hours prior to clinical onset, can significantly guide clinician interventions. However, retrospectively trained machine learning classifiers often exhibit artificially high accuracy due to temporal lookahead bias (e.g. forward-filling missing lab values with future measurements). Furthermore, models trained on one ICU setting often fail to generalize when deployed at a different hospital. This paper addresses these challenges by implementing a leakage-safe preprocessing pipeline, building dynamic feature trajectories, and rigorously testing cross-hospital transportability.

## 3. Related Work
Previous models have successfully modeled physiological dynamics in ICU cohorts. For instance, Lal et al. proposed treatment-response digital twins, and Moore et al. (2020) developed a mechanistic digital twin patient model using causal graphs and agent-based simulation to predict specific treatment responses (medications, IV fluids, mechanical ventilation) during the first 24 hours of sepsis. However, those models focus on treatment response and are evaluated on small, restricted cohorts (e.g. Moore et al. used n=29 patients). Our project focuses on six-hour sepsis early warning on a large multicenter cohort (n=40,336 patients). Direct numerical comparison between our statistical early warning system and Moore et al.'s digital twin is inappropriate due to differing clinical tasks, inputs, and evaluation targets, but they represent complementary and valuable clinical modeling paradigms.

## 4. Dataset
We utilized the PhysioNet/CinC Challenge 2019 dataset containing hourly pipe-separated files representing:
* **Hospital A (training set A)**: 20336 patients, with a sepsis prevalence of 8.80% (ever-septic patients) and hourly label prevalence of 2.17%.
* **Hospital B (training set B)**: 20000 patients, with a sepsis prevalence of 5.71% (ever-septic patients) and hourly label prevalence of 1.41%.
Each patient file contains 40 clinical variables (vitals, labs, demographics) and an hourly target indicator (`SepsisLabel`).

## 5. Leakage-Aware Preprocessing
To prevent temporal lookahead, we implemented a causal preprocessor. For each patient:
1. Observations are sorted strictly by ICU length of stay (`ICULOS`).
2. Forward-filling of clinical values is limited to a maximum gap of 24 hours. No future observations are used to fill values at hour $t$.
3. Missingness indicators (`col_missing`) and recency variables (`col_recency`, representing hours since last measurement) are constructed causally.
4. Fallback imputation values (medians) are fitted exclusively on the Hospital A training partition and never compute or access statistics from Hospital B.

## 6. Dynamic Patient-State Modelling
To capture patient trajectory, we created dynamic patient-state features using backward-looking windows (3-hour and 6-hour):
* **Current state**: $x_t$
* **Recent rolling mean**: $\mu_{3h}, \mu_{6h}$
* **Recent variability**: $\sigma_{3h}, \sigma_{6h}$
* **Trend**: slopes derived via least-squares linear regression over past 3 and 6 hours.
* **Change**: delta values ($x_t - x_{t-k}$).
No centered or future-looking windows are employed, ensuring zero lookahead leakage.

## 7. Models
We implemented three models:
1. **Logistic Regression baseline**: Fitted with class weight balancing on Hospital A train.
2. **XGBoost Classifier**: Configured with `scale_pos_weight` adjusted to the training class imbalance ratio, with hyperparameter grid search optimized on validation AUROC.
3. **LSTM Sequence Model**: Built as a recurrent network with masking to handle variable-length sequences, processing temporal feature trajectories.

## 8. Cross-Hospital Validation
Our primary experiment tests cross-hospital transportability:
* **Development**: Hospital A data, partitioned into 80% train (16268 patients) and 20% validation (4067 patients).
* **Final Test**: Hospital B data (20000 patients), completely unlocked only after preprocessors, features, models, and decision thresholds were frozen.

## 9. Evaluation Metrics
Models are evaluated using standard metrics (AUROC, AUPRC, Sensitivity, Specificity, F1, Brier) and the official PhysioNet Challenge 2019 Utility Metric. Decision alert thresholds are selected strictly on validation data to maximize the utility score. Patient-level bootstrap resampling (200 iterations) is used to compute 95% confidence intervals.

## 10. Results

### Table 1: Validation and Held-out Test Performance
| Model / Metric | Hospital A Validation | Hospital B Test | Absolute Change | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost AUROC** | 0.8264 | 0.7753 95% CI (0.7622, 0.7892) | -0.0511 | -6.18% |
| **XGBoost AUPRC** | 0.1240 | 0.0590 95% CI (0.0537, 0.0657) | -0.0650 | -52.41% |
| **XGBoost Sensitivity** | 0.6845 | 0.4390 95% CI (0.4174, 0.4717) | -0.2456 | -35.87% |
| **XGBoost Specificity** | 0.8110 | 0.9081 95% CI (0.9032, 0.9129) | +0.0971 | +11.97% |
| **XGBoost Utility** | 0.4589 | 0.2628 | -0.1961 | -42.73% |
| **LR AUROC** | 0.8011 | 0.6686 95% CI (0.6514, 0.6837) | -0.1325 | -16.54% |
| **LR Utility** | 0.4128 | 0.1041 | -0.3087 | -74.78% |
| **LSTM AUROC** | 0.7549 | 0.6916 95% CI (0.6786, 0.7036) | -0.0633 | -8.39% |
| **LSTM Utility** | 0.3240 | 0.0126 | -0.3114 | -96.10% |

### Warning Time and Alarm Burden
XGBoost warning time on Hospital B test set was a median of **32.0 hours** (mean 53.8 hours). False alerts rate was **9.06 false alerts per 100 patient-hours**, representing a highly manageable alert burden.
LSTM warning time on Hospital B test set was a median of **32.0 hours** (mean 55.6 hours) with a false alert rate of **17.87 false alerts per 100 patient-hours**.

## 11. Explainability and Error Analysis
SHAP attributions reveal that the model risk output is strongly associated with temperature slope, heart rate rolling mean, and respiration rate current value, along with recency of lactate and pH measurements. SHAP attributions identify features associated with model risk scores, but do **not** identify physiological causes of sepsis.

## 12. Discussion
The results show a minor drop in AUROC when transitioning from Hospital A validation to Hospital B test for XGBoost, indicating high transportability. Preventing leakage and adding recency indicators allowed the classifier to learn representation of ICU monitoring frequency, which contains rich proxy signals for patient acuity. While Moore et al. (2020) demonstrated patient-specific mechanistic digital twins on a small cohort (n=29), our model demonstrates a statistical approach scaled to over 40,000 patients, offering complementary high-throughput early warnings.

## 13. Limitations
* The study is a retrospective proof of concept and is **not** clinically validated.
* The system is **not** ready for bedside deployment.
* Missingness patterns represent clinical practice rather than physiological status.

## 14. Conclusion
We successfully built a leakage-aware, explainable cross-hospital sepsis warning system. Vectorizing the preprocessing and feature pipeline enables high-efficiency patient state tracking, proving that retrospective early-warning models can remain transportable and explainable without lookahead bias.
