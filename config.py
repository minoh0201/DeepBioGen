# File names
MARKER_T2D = "marker_T2D.txt"
MARKER_WT2D = "marker_WT2D.txt"

ICB_TRAIN_MAT = ["ICB_Hugo_mat.csv", "ICB_Riaz_mat.csv"]
ICB_TRAIN_CLS = ["ICB_Hugo_cls.csv", "ICB_Riaz_cls.csv"]

ICB_TEST_MAT = "ICB_Gide_mat.csv"
ICB_TEST_CLS = "ICB_Gide_cls.csv"

ICB_COMB_TEST_MAT = "ICB_Gide_comb_mat.csv"
ICB_COMB_TEST_CLS = "ICB_Gide_comb_cls.csv"

ICB_TCELL_SIG = "Tcell_signatures.csv"

# Hyper-parameter grid for training classifiers
svm_hyper_parameters = [{'C': [2 ** s for s in range(-4, 4, 1)], 'kernel': ['linear']},
                        {'C': [2 ** s for s in range(-4, 4, 1)], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}]
rf_hyper_parameters = [{'n_estimators': [2 ** s for s in range(7, 11, 1)],
                        'max_features': ['sqrt', 'log2'],
                        'criterion': ['gini', 'entropy']
                        }, ]
mlp_hyper_parameters = [{'hidden_layer_sizes': [(128, 64, 32), (128, 64, 32, 16), (128, 64, 32, 16, 8)],
                            'learning_rate': ['constant', 'invscaling', 'adaptive'],
                            'alpha': [10** s for s in range(-4, 0, 1)]
                            }]
