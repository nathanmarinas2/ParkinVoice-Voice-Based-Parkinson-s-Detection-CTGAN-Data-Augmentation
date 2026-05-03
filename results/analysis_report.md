# Parkinson voice classification analysis

## Dataset summary

- Samples: 756
- Predictive variables after removing `id`: 753
- Target distribution: class 1 = 564, class 0 = 192
- Final hold-out split: 604 train / 152 test

## Methodology

1. A stratified train/test split was created before any feature selection or model fitting.
2. Feature relevance was ranked on the real training split using a combined score: mutual information + RandomForest importance.
3. The number of retained features was chosen with 5-fold CV and balanced accuracy.
4. Five classical models were tuned on the reduced feature space.
5. CTGAN was then fitted only on the selected features of the training split plus the target, and a purely synthetic training set with the same class counts was generated.
6. Both strategies were evaluated on the same real hold-out test split.

## Selected feature subset

- Selected feature count: 40
- Selected features: std_delta_delta_log_energy, std_delta_log_energy, std_6th_delta_delta, tqwt_entropy_log_dec_12, std_8th_delta_delta, tqwt_TKEO_std_dec_13, std_9th_delta_delta, tqwt_TKEO_mean_dec_12, std_7th_delta_delta, std_6th_delta, mean_MFCC_2nd_coef, tqwt_entropy_shannon_dec_13, tqwt_minValue_dec_13, tqwt_TKEO_mean_dec_13, tqwt_stdValue_dec_12, rapJitter, std_8th_delta, tqwt_entropy_log_dec_16, tqwt_entropy_log_dec_13, tqwt_entropy_log_dec_35, tqwt_stdValue_dec_13, tqwt_TKEO_std_dec_12, tqwt_energy_dec_27, std_7th_delta, minIntensity, tqwt_energy_dec_26, tqwt_kurtosisValue_dec_18, apq11Shimmer, tqwt_entropy_log_dec_11, tqwt_maxValue_dec_12, tqwt_entropy_shannon_dec_12, std_10th_delta_delta, tqwt_medianValue_dec_8, tqwt_entropy_shannon_dec_14, tqwt_TKEO_mean_dec_16, tqwt_medianValue_dec_26, tqwt_entropy_log_dec_27, tqwt_TKEO_std_dec_7, tqwt_stdValue_dec_15, locAbsJitter

![Feature ranking](figures/feature_ranking.png)

![Feature count selection](figures/feature_count_selection.png)

## Real-data strategy

- Best CV model: RandomForest
- Hold-out balanced accuracy: 0.7718
- Hold-out recall: 0.9027
- Hold-out F1: 0.8908
- Hold-out ROC AUC: 0.8870

| strategy   | model              |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_f1_mean |   cv_roc_auc_mean |   test_accuracy |   test_balanced_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc | best_params                                                                        |
|:-----------|:-------------------|----------------------------:|---------------------------:|-------------:|------------------:|----------------:|-------------------------:|-----------------:|--------------:|----------:|---------------:|:-----------------------------------------------------------------------------------|
| real       | RandomForest       |                      0.8049 |                     0.0469 |       0.9145 |            0.9071 |          0.8355 |                   0.7718 |           0.8793 |        0.9027 |    0.8908 |         0.887  | {"max_depth": 12, "max_features": 0.3, "min_samples_leaf": 4, "n_estimators": 300} |
| real       | GradientBoosting   |                      0.8038 |                     0.0272 |       0.9183 |            0.9062 |          0.875  |                   0.7984 |           0.8852 |        0.9558 |    0.9191 |         0.916  | {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}                        |
| real       | KNN                |                      0.7959 |                     0.0292 |       0.9126 |            0.8656 |          0.8487 |                   0.8059 |           0.9018 |        0.8938 |    0.8978 |         0.8845 | {"model__n_neighbors": 3, "model__p": 1, "model__weights": "uniform"}              |
| real       | SVM                |                      0.7866 |                     0.0232 |       0.8757 |            0.8943 |          0.8421 |                   0.8182 |           0.9159 |        0.8673 |    0.8909 |         0.9099 | {"model__C": 3.0, "model__gamma": "scale", "model__kernel": "rbf"}                 |
| real       | LogisticRegression |                      0.7776 |                     0.0309 |       0.8505 |            0.8779 |          0.7763 |                   0.7656 |           0.899  |        0.7876 |    0.8396 |         0.8223 | {"model__C": 0.1}                                                                  |

## CTGAN synthetic-data strategy

- CTGAN epochs: 150
- Best CV model: SVM
- Hold-out balanced accuracy: 0.6591
- Hold-out recall: 0.8053
- Hold-out F1: 0.8161
- Hold-out ROC AUC: 0.7243

| strategy   | model              |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_f1_mean |   cv_roc_auc_mean |   test_accuracy |   test_balanced_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc | best_params                                                                          |
|:-----------|:-------------------|----------------------------:|---------------------------:|-------------:|------------------:|----------------:|-------------------------:|-----------------:|--------------:|----------:|---------------:|:-------------------------------------------------------------------------------------|
| synthetic  | SVM                |                      0.5375 |                     0.0398 |       0.5991 |            0.5478 |          0.7303 |                   0.6591 |           0.8273 |        0.8053 |    0.8161 |         0.7243 | {"model__C": 3.0, "model__gamma": 0.001, "model__kernel": "rbf"}                     |
| synthetic  | KNN                |                      0.5339 |                     0.0249 |       0.7438 |            0.5014 |          0.7105 |                   0.5534 |           0.7674 |        0.8761 |    0.8182 |         0.5015 | {"model__n_neighbors": 5, "model__p": 2, "model__weights": "uniform"}                |
| synthetic  | LogisticRegression |                      0.5244 |                     0.0506 |       0.6137 |            0.5276 |          0.7237 |                   0.6714 |           0.8381 |        0.7788 |    0.8073 |         0.7279 | {"model__C": 0.1}                                                                    |
| synthetic  | RandomForest       |                      0.5196 |                     0.0244 |       0.7879 |            0.5293 |          0.7434 |                   0.5    |           0.7434 |        1      |    0.8528 |         0.5006 | {"max_depth": null, "max_features": 0.3, "min_samples_leaf": 4, "n_estimators": 300} |
| synthetic  | GradientBoosting   |                      0.5115 |                     0.0103 |       0.7608 |            0.5062 |          0.6842 |                   0.5525 |           0.7686 |        0.823  |    0.7949 |         0.59   | {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100}                          |

## Comparison

- Balanced-accuracy gap (real - synthetic): 0.1128
- Recall gap (real - synthetic): 0.0973
- F1 gap (real - synthetic): 0.0747
- ROC AUC gap (real - synthetic): 0.1627

The real-data strategy should be treated as the deployment reference if the synthetic strategy underperforms on recall or balanced accuracy. The synthetic model is still useful to discuss privacy-preserving augmentation, stress testing, and how much predictive signal CTGAN preserves after compression to the selected feature space.

![Model comparison](figures/model_comparison.png)

![ROC curves](figures/roc_curves.png)

![Confusion matrices](figures/confusion_matrices.png)

## Synthetic data quality

- Mean KS statistic across selected features: 0.4058
- Max KS statistic: 0.8742
- Mean normalized Wasserstein distance: 0.8977
- Correlation-matrix MAE: 0.3094

The drift values below indicate which selected variables are hardest for CTGAN to reproduce faithfully. They are the best candidates to discuss when explaining why the synthetic classifier may lose sensitivity.

| feature                |   ks_stat |   wasserstein_normalized |
|:-----------------------|----------:|-------------------------:|
| locAbsJitter           |    0.8742 |                   8.5047 |
| tqwt_TKEO_std_dec_7    |    0.7947 |                   0.2359 |
| tqwt_medianValue_dec_8 |    0.7666 |                   5.588  |
| tqwt_energy_dec_27     |    0.6788 |                   0.3426 |
| std_8th_delta_delta    |    0.6424 |                   1.279  |

![Synthetic quality](figures/synthetic_quality.png)

## Conclusion

The complete comparison is now reproducible from a single script. If the real-data model wins by a clear margin, the interpretation is straightforward: synthetic data preserves part of the decision boundary, but not all of it. If the gap is small, you can argue that the selected voice descriptors retain enough structure for CTGAN to train a clinically useful proxy model.
