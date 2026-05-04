window.APP_DATA = {
  "project": {
    "title": "ParkinVoice: Parkinson detection from voice descriptors",
    "subtitle": "Comparative analysis between classical models trained on real data, purely synthetic CTGAN data, and a mixed scenario."
  },
  "dataset": {
    "rows": 756,
    "features": 753,
    "class_counts": {
      "0": 192,
      "1": 564
    },
    "train_rows": 604,
    "test_rows": 152
  },
  "methodology": [
    "Estratificar train/test antes de cualquier selecci\u00f3n de variables.",
    "Rankear variables con informaci\u00f3n mutua e importancia de RandomForest.",
    "Seleccionar un subconjunto compacto de rasgos vocales.",
    "Comparar entrenamiento con datos reales, sint\u00e9ticos y mixtos sobre el mismo test real.",
    "Analizar deriva de CTGAN y estabilidad de los resultados con varias semillas."
  ],
  "figures": {
    "featureRanking": "../results/figures/feature_ranking.png",
    "featureCount": "../results/figures/feature_count_selection.png",
    "featureFamilies": "../results/figures/feature_family_summary.png",
    "modelComparison": "../results/figures/model_comparison.png",
    "rocCurves": "../results/figures/roc_curves.png",
    "confusionMatrices": "../results/figures/confusion_matrices.png",
    "syntheticQuality": "../results/figures/synthetic_quality.png",
    "robustness": "../results/figures/robustness_summary.png"
  },
  "selectedFeatureCount": 40,
  "topFeatures": [
    {
      "feature": "std_delta_delta_log_energy",
      "combined_score": 1.0,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "std_delta_log_energy",
      "combined_score": 0.9367,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "std_6th_delta_delta",
      "combined_score": 0.777,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "tqwt_entropy_log_dec_12",
      "combined_score": 0.7383,
      "family": "TQWT / wavelet"
    },
    {
      "feature": "std_8th_delta_delta",
      "combined_score": 0.6973,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "tqwt_TKEO_std_dec_13",
      "combined_score": 0.6916,
      "family": "TQWT / wavelet"
    },
    {
      "feature": "std_9th_delta_delta",
      "combined_score": 0.6809,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "tqwt_TKEO_mean_dec_12",
      "combined_score": 0.6657,
      "family": "TQWT / wavelet"
    },
    {
      "feature": "std_7th_delta_delta",
      "combined_score": 0.6601,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "std_6th_delta",
      "combined_score": 0.6583,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "mean_MFCC_2nd_coef",
      "combined_score": 0.6362,
      "family": "MFCC / energ\u00eda / din\u00e1micas"
    },
    {
      "feature": "tqwt_entropy_shannon_dec_13",
      "combined_score": 0.607,
      "family": "TQWT / wavelet"
    }
  ],
  "featureFamilies": [
    {
      "family": "TQWT / wavelet",
      "count": 25,
      "share": 0.625,
      "mean_combined_score": 0.5501,
      "description": "Describe patrones multiescala en la voz; captura irregularidades temporales y espectrales que suelen aparecer cuando la fonaci\u00f3n pierde estabilidad.",
      "features": "tqwt_entropy_log_dec_12, tqwt_TKEO_std_dec_13, tqwt_TKEO_mean_dec_12, tqwt_entropy_shannon_dec_13, tqwt_minValue_dec_13, tqwt_TKEO_mean_dec_13, tqwt_stdValue_dec_12, tqwt_entropy_log_dec_16, tqwt_entropy_log_dec_13, tqwt_entropy_log_dec_35, tqwt_stdValue_dec_13, tqwt_TKEO_std_dec_12, tqwt_energy_dec_27, tqwt_energy_dec_26, tqwt_kurtosisValue_dec_18, tqwt_entropy_log_dec_11, tqwt_maxValue_dec_12, tqwt_entropy_shannon_dec_12, tqwt_medianValue_dec_8, tqwt_entropy_shannon_dec_14, tqwt_TKEO_mean_dec_16, tqwt_medianValue_dec_26, tqwt_entropy_log_dec_27, tqwt_TKEO_std_dec_7, tqwt_stdValue_dec_15"
    },
    {
      "family": "MFCC / energ\u00eda / din\u00e1micas",
      "count": 11,
      "share": 0.275,
      "mean_combined_score": 0.6981,
      "description": "Resume la envolvente espectral y su evoluci\u00f3n temporal; refleja cambios finos en timbre, articulaci\u00f3n y control motor de la voz.",
      "features": "std_delta_delta_log_energy, std_delta_log_energy, std_6th_delta_delta, std_8th_delta_delta, std_9th_delta_delta, std_7th_delta_delta, std_6th_delta, mean_MFCC_2nd_coef, std_8th_delta, std_7th_delta, std_10th_delta_delta"
    },
    {
      "family": "Jitter / shimmer",
      "count": 3,
      "share": 0.075,
      "mean_combined_score": 0.5221,
      "description": "Mide microvariaciones ciclo a ciclo en frecuencia y amplitud; se asocia con inestabilidad de los pliegues vocales.",
      "features": "rapJitter, apq11Shimmer, locAbsJitter"
    },
    {
      "family": "Intensidad",
      "count": 1,
      "share": 0.025,
      "mean_combined_score": 0.5478,
      "description": "Caracteriza la energ\u00eda y la proyecci\u00f3n de la se\u00f1al, \u00fatil para detectar alteraciones globales en la emisi\u00f3n vocal.",
      "features": "minIntensity"
    }
  ],
  "strategies": {
    "real": {
      "selected_model_by_cv": "RandomForest",
      "selected_metrics": {
        "accuracy": 0.8355,
        "balanced_accuracy": 0.7718,
        "precision": 0.8793,
        "sensitivity": 0.9027,
        "specificity": 0.641,
        "f1": 0.8908,
        "roc_auc": 0.887
      },
      "holdout_best_model": "SVM",
      "holdout_best_metrics": {
        "accuracy": 0.8421,
        "balanced_accuracy": 0.8182,
        "precision": 0.9159,
        "sensitivity": 0.8673,
        "specificity": 0.7692,
        "f1": 0.8909,
        "roc_auc": 0.9099
      },
      "summary_table": [
        {
          "strategy": "real",
          "model": "RandomForest",
          "cv_balanced_accuracy_mean": 0.8049,
          "cv_balanced_accuracy_std": 0.0469,
          "cv_f1_mean": 0.9145,
          "cv_roc_auc_mean": 0.9071,
          "test_accuracy": 0.8355,
          "test_balanced_accuracy": 0.7718,
          "test_precision": 0.8793,
          "test_sensitivity": 0.9027,
          "test_specificity": 0.641,
          "test_f1": 0.8908,
          "test_roc_auc": 0.887,
          "best_params": "{\"max_depth\": 12, \"max_features\": 0.3, \"min_samples_leaf\": 4, \"n_estimators\": 300}"
        },
        {
          "strategy": "real",
          "model": "GradientBoosting",
          "cv_balanced_accuracy_mean": 0.8038,
          "cv_balanced_accuracy_std": 0.0272,
          "cv_f1_mean": 0.9183,
          "cv_roc_auc_mean": 0.9062,
          "test_accuracy": 0.875,
          "test_balanced_accuracy": 0.7984,
          "test_precision": 0.8852,
          "test_sensitivity": 0.9558,
          "test_specificity": 0.641,
          "test_f1": 0.9191,
          "test_roc_auc": 0.916,
          "best_params": "{\"learning_rate\": 0.1, \"max_depth\": 3, \"n_estimators\": 200}"
        },
        {
          "strategy": "real",
          "model": "KNN",
          "cv_balanced_accuracy_mean": 0.7959,
          "cv_balanced_accuracy_std": 0.0292,
          "cv_f1_mean": 0.9126,
          "cv_roc_auc_mean": 0.8656,
          "test_accuracy": 0.8487,
          "test_balanced_accuracy": 0.8059,
          "test_precision": 0.9018,
          "test_sensitivity": 0.8938,
          "test_specificity": 0.7179,
          "test_f1": 0.8978,
          "test_roc_auc": 0.8845,
          "best_params": "{\"model__n_neighbors\": 3, \"model__p\": 1, \"model__weights\": \"uniform\"}"
        },
        {
          "strategy": "real",
          "model": "SVM",
          "cv_balanced_accuracy_mean": 0.7866,
          "cv_balanced_accuracy_std": 0.0232,
          "cv_f1_mean": 0.8757,
          "cv_roc_auc_mean": 0.8943,
          "test_accuracy": 0.8421,
          "test_balanced_accuracy": 0.8182,
          "test_precision": 0.9159,
          "test_sensitivity": 0.8673,
          "test_specificity": 0.7692,
          "test_f1": 0.8909,
          "test_roc_auc": 0.9099,
          "best_params": "{\"model__C\": 3.0, \"model__gamma\": \"scale\", \"model__kernel\": \"rbf\"}"
        },
        {
          "strategy": "real",
          "model": "LogisticRegression",
          "cv_balanced_accuracy_mean": 0.7776,
          "cv_balanced_accuracy_std": 0.0309,
          "cv_f1_mean": 0.8505,
          "cv_roc_auc_mean": 0.8779,
          "test_accuracy": 0.7763,
          "test_balanced_accuracy": 0.7656,
          "test_precision": 0.899,
          "test_sensitivity": 0.7876,
          "test_specificity": 0.7436,
          "test_f1": 0.8396,
          "test_roc_auc": 0.8223,
          "best_params": "{\"model__C\": 0.1}"
        }
      ]
    },
    "synthetic": {
      "selected_model_by_cv": "KNN",
      "selected_metrics": {
        "accuracy": 0.6579,
        "balanced_accuracy": 0.5936,
        "precision": 0.7961,
        "sensitivity": 0.7257,
        "specificity": 0.4615,
        "f1": 0.7593,
        "roc_auc": 0.6054
      },
      "holdout_best_model": "SVM",
      "holdout_best_metrics": {
        "accuracy": 0.5592,
        "balanced_accuracy": 0.6112,
        "precision": 0.8382,
        "sensitivity": 0.5044,
        "specificity": 0.7179,
        "f1": 0.6298,
        "roc_auc": 0.3749
      },
      "summary_table": [
        {
          "strategy": "synthetic",
          "model": "KNN",
          "cv_balanced_accuracy_mean": 0.5317,
          "cv_balanced_accuracy_std": 0.0346,
          "cv_f1_mean": 0.6378,
          "cv_roc_auc_mean": 0.5371,
          "test_accuracy": 0.6579,
          "test_balanced_accuracy": 0.5936,
          "test_precision": 0.7961,
          "test_sensitivity": 0.7257,
          "test_specificity": 0.4615,
          "test_f1": 0.7593,
          "test_roc_auc": 0.6054,
          "best_params": "{\"model__n_neighbors\": 5, \"model__p\": 1, \"model__weights\": \"uniform\"}"
        },
        {
          "strategy": "synthetic",
          "model": "RandomForest",
          "cv_balanced_accuracy_mean": 0.5126,
          "cv_balanced_accuracy_std": 0.0114,
          "cv_f1_mean": 0.6733,
          "cv_roc_auc_mean": 0.5115,
          "test_accuracy": 0.6776,
          "test_balanced_accuracy": 0.5985,
          "test_precision": 0.7963,
          "test_sensitivity": 0.7611,
          "test_specificity": 0.4359,
          "test_f1": 0.7783,
          "test_roc_auc": 0.6506,
          "best_params": "{\"max_depth\": 12, \"max_features\": \"sqrt\", \"min_samples_leaf\": 4, \"n_estimators\": 500}"
        },
        {
          "strategy": "synthetic",
          "model": "SVM",
          "cv_balanced_accuracy_mean": 0.508,
          "cv_balanced_accuracy_std": 0.0227,
          "cv_f1_mean": 0.5843,
          "cv_roc_auc_mean": 0.5042,
          "test_accuracy": 0.5592,
          "test_balanced_accuracy": 0.6112,
          "test_precision": 0.8382,
          "test_sensitivity": 0.5044,
          "test_specificity": 0.7179,
          "test_f1": 0.6298,
          "test_roc_auc": 0.3749,
          "best_params": "{\"model__C\": 1.0, \"model__gamma\": 0.01, \"model__kernel\": \"rbf\"}"
        },
        {
          "strategy": "synthetic",
          "model": "GradientBoosting",
          "cv_balanced_accuracy_mean": 0.5009,
          "cv_balanced_accuracy_std": 0.0585,
          "cv_f1_mean": 0.6226,
          "cv_roc_auc_mean": 0.4916,
          "test_accuracy": 0.5526,
          "test_balanced_accuracy": 0.5396,
          "test_precision": 0.7711,
          "test_sensitivity": 0.5664,
          "test_specificity": 0.5128,
          "test_f1": 0.6531,
          "test_roc_auc": 0.5373,
          "best_params": "{\"learning_rate\": 0.1, \"max_depth\": 2, \"n_estimators\": 200}"
        },
        {
          "strategy": "synthetic",
          "model": "LogisticRegression",
          "cv_balanced_accuracy_mean": 0.4933,
          "cv_balanced_accuracy_std": 0.0328,
          "cv_f1_mean": 0.5591,
          "cv_roc_auc_mean": 0.485,
          "test_accuracy": 0.5921,
          "test_balanced_accuracy": 0.5997,
          "test_precision": 0.8148,
          "test_sensitivity": 0.5841,
          "test_specificity": 0.6154,
          "test_f1": 0.6804,
          "test_roc_auc": 0.6274,
          "best_params": "{\"model__C\": 0.1}"
        }
      ]
    },
    "mixed": {
      "selected_model_by_cv": "SVM",
      "selected_metrics": {
        "accuracy": 0.7763,
        "balanced_accuracy": 0.7068,
        "precision": 0.8496,
        "sensitivity": 0.8496,
        "specificity": 0.5641,
        "f1": 0.8496,
        "roc_auc": 0.8169
      },
      "holdout_best_model": "KNN",
      "holdout_best_metrics": {
        "accuracy": 0.8553,
        "balanced_accuracy": 0.8103,
        "precision": 0.9027,
        "sensitivity": 0.9027,
        "specificity": 0.7179,
        "f1": 0.9027,
        "roc_auc": 0.8885
      },
      "summary_table": [
        {
          "strategy": "mixed",
          "model": "SVM",
          "cv_balanced_accuracy_mean": 0.6684,
          "cv_balanced_accuracy_std": 0.0164,
          "cv_f1_mean": 0.666,
          "cv_roc_auc_mean": 0.723,
          "test_accuracy": 0.7763,
          "test_balanced_accuracy": 0.7068,
          "test_precision": 0.8496,
          "test_sensitivity": 0.8496,
          "test_specificity": 0.5641,
          "test_f1": 0.8496,
          "test_roc_auc": 0.8169,
          "best_params": "{\"model__C\": 0.5, \"model__gamma\": \"scale\", \"model__kernel\": \"rbf\"}"
        },
        {
          "strategy": "mixed",
          "model": "RandomForest",
          "cv_balanced_accuracy_mean": 0.6437,
          "cv_balanced_accuracy_std": 0.0241,
          "cv_f1_mean": 0.7911,
          "cv_roc_auc_mean": 0.7568,
          "test_accuracy": 0.8355,
          "test_balanced_accuracy": 0.755,
          "test_precision": 0.8667,
          "test_sensitivity": 0.9204,
          "test_specificity": 0.5897,
          "test_f1": 0.8927,
          "test_roc_auc": 0.8829,
          "best_params": "{\"max_depth\": 12, \"max_features\": 0.3, \"min_samples_leaf\": 4, \"n_estimators\": 500}"
        },
        {
          "strategy": "mixed",
          "model": "LogisticRegression",
          "cv_balanced_accuracy_mean": 0.6336,
          "cv_balanced_accuracy_std": 0.0174,
          "cv_f1_mean": 0.6953,
          "cv_roc_auc_mean": 0.7125,
          "test_accuracy": 0.7632,
          "test_balanced_accuracy": 0.7064,
          "test_precision": 0.8532,
          "test_sensitivity": 0.823,
          "test_specificity": 0.5897,
          "test_f1": 0.8378,
          "test_roc_auc": 0.8205,
          "best_params": "{\"model__C\": 0.1}"
        },
        {
          "strategy": "mixed",
          "model": "KNN",
          "cv_balanced_accuracy_mean": 0.633,
          "cv_balanced_accuracy_std": 0.0173,
          "cv_f1_mean": 0.7799,
          "cv_roc_auc_mean": 0.6865,
          "test_accuracy": 0.8553,
          "test_balanced_accuracy": 0.8103,
          "test_precision": 0.9027,
          "test_sensitivity": 0.9027,
          "test_specificity": 0.7179,
          "test_f1": 0.9027,
          "test_roc_auc": 0.8885,
          "best_params": "{\"model__n_neighbors\": 3, \"model__p\": 1, \"model__weights\": \"uniform\"}"
        },
        {
          "strategy": "mixed",
          "model": "GradientBoosting",
          "cv_balanced_accuracy_mean": 0.6177,
          "cv_balanced_accuracy_std": 0.0256,
          "cv_f1_mean": 0.7776,
          "cv_roc_auc_mean": 0.7346,
          "test_accuracy": 0.7697,
          "test_balanced_accuracy": 0.6268,
          "test_precision": 0.8,
          "test_sensitivity": 0.9204,
          "test_specificity": 0.3333,
          "test_f1": 0.856,
          "test_roc_auc": 0.8312,
          "best_params": "{\"learning_rate\": 0.1, \"max_depth\": 2, \"n_estimators\": 200}"
        }
      ]
    }
  },
  "syntheticQuality": {
    "summary": {
      "mean_ks_stat": 0.304,
      "max_ks_stat": 0.8477,
      "mean_normalized_wasserstein": 0.5976,
      "correlation_mae": 0.3134
    },
    "topDrift": [
      {
        "feature": "locAbsJitter",
        "ks_stat": 0.8477,
        "wasserstein_normalized": 1.3429,
        "family": "Jitter / shimmer"
      },
      {
        "feature": "rapJitter",
        "ks_stat": 0.7881,
        "wasserstein_normalized": 0.3896,
        "family": "Jitter / shimmer"
      },
      {
        "feature": "tqwt_TKEO_std_dec_12",
        "ks_stat": 0.606,
        "wasserstein_normalized": 0.0852,
        "family": "TQWT / wavelet"
      },
      {
        "feature": "tqwt_TKEO_mean_dec_12",
        "ks_stat": 0.5546,
        "wasserstein_normalized": 0.136,
        "family": "TQWT / wavelet"
      },
      {
        "feature": "tqwt_TKEO_mean_dec_13",
        "ks_stat": 0.548,
        "wasserstein_normalized": 0.2298,
        "family": "TQWT / wavelet"
      },
      {
        "feature": "tqwt_medianValue_dec_8",
        "ks_stat": 0.4851,
        "wasserstein_normalized": 7.3965,
        "family": "TQWT / wavelet"
      }
    ]
  },
  "robustness": {
    "seeds": [
      13,
      42
    ],
    "ctgan_epochs": 25,
    "summary": [
      {
        "strategy": "real",
        "model": "RandomForest",
        "accuracy_mean": 0.8224,
        "accuracy_std": 0.0132,
        "balanced_accuracy_mean": 0.7504,
        "balanced_accuracy_std": 0.0214,
        "precision_mean": 0.8676,
        "precision_std": 0.0117,
        "sensitivity_mean": 0.8982,
        "sensitivity_std": 0.0044,
        "specificity_mean": 0.6026,
        "specificity_std": 0.0385,
        "f1_mean": 0.8826,
        "f1_std": 0.0082,
        "roc_auc_mean": 0.8769,
        "roc_auc_std": 0.0101
      },
      {
        "strategy": "mixed",
        "model": "SVM",
        "accuracy_mean": 0.7961,
        "accuracy_std": 0.0066,
        "balanced_accuracy_mean": 0.7201,
        "balanced_accuracy_std": 0.0212,
        "precision_mean": 0.8538,
        "precision_std": 0.0135,
        "sensitivity_mean": 0.8761,
        "sensitivity_std": 0.0088,
        "specificity_mean": 0.5641,
        "specificity_std": 0.0513,
        "f1_mean": 0.8647,
        "f1_std": 0.0026,
        "roc_auc_mean": 0.8266,
        "roc_auc_std": 0.005
      },
      {
        "strategy": "synthetic",
        "model": "KNN",
        "accuracy_mean": 0.5757,
        "accuracy_std": 0.0625,
        "balanced_accuracy_mean": 0.4627,
        "balanced_accuracy_std": 0.084,
        "precision_mean": 0.7237,
        "precision_std": 0.0448,
        "sensitivity_mean": 0.6947,
        "sensitivity_std": 0.0398,
        "specificity_mean": 0.2308,
        "specificity_std": 0.1282,
        "f1_mean": 0.7089,
        "f1_std": 0.0422,
        "roc_auc_mean": 0.4781,
        "roc_auc_std": 0.1078
      }
    ]
  },
  "conclusions": [
    "Los datos reales siguen siendo la referencia de despliegue: maximizan balanced accuracy y ROC AUC.",
    "CTGAN conserva parte de la se\u00f1al discriminativa, pero no reemplaza al entrenamiento real en este dataset.",
    "La familia TQWT domina la selecci\u00f3n, lo que sugiere que la informaci\u00f3n multiescala es cr\u00edtica para detectar Parkinson en la voz.",
    "La deriva sint\u00e9tica m\u00e1s fuerte aparece en rasgos de jitter y en varias descomposiciones TQWT, coherente con la ca\u00edda de rendimiento.",
    "El escenario mixto responde a la pregunta pr\u00e1ctica importante: el sint\u00e9tico puro empeora, pero puede seguir siendo \u00fatil como complemento experimental."
  ]
};
