import os
import time
import numpy as np
import pandas as pd

import config
import pickle
from experimentor import DataContainer

def load_T2D(train_data_filename=config.MARKER_T2D, test_data_filename=config.MARKER_WT2D) -> DataContainer:
    # Time stamp
    start_time = time.time()

    cache_filename = os.path.join(os.getcwd(), 'data', train_data_filename + "-" + test_data_filename + ".pkl")

    # If not cached, cache data
    if not os.path.exists(cache_filename):
        # Indicators
        feature_string = "gi|"
        label_string = "disease"
        label_dict = {
                # Controls
                'n': 0,
                # T2D patients in both T2D and WT2D data
                't2d': 1,
            }

        train_data_filepath = os.path.join(os.getcwd(), 'data', train_data_filename)
        test_data_filepath = os.path.join(os.getcwd(), 'data', test_data_filename)

        # Read files
        if os.path.isfile(train_data_filepath) and os.path.isfile(test_data_filepath):
            raw1 = pd.read_csv(train_data_filepath, sep='\t', index_col=0, header=None, dtype='str')
            raw2 = pd.read_csv(test_data_filepath, sep='\t', index_col=0, header=None, dtype='str')
        else:
            print("FileNotFoundError: one of files does not exist")
            exit()
            
        # Select rows having feature index identifier string
        X1 = raw1.loc[raw1.index.str.contains(feature_string, regex=False)].T
        X2 = raw2.loc[raw2.index.str.contains(feature_string, regex=False)].T

        IntersectionIndex = X1.columns.intersection(X2.columns)

        X1 = X1.loc[:,IntersectionIndex].values.astype(np.float64)
        X2 = X2.loc[:,IntersectionIndex].values.astype(np.float64)

        # Get class labels
        y1 = raw1.loc[label_string]
        y1 = y1.replace(label_dict)
        y1 = y1.values.astype(np.float64)

        y2 = raw2.loc[label_string]
        y2 = y2.replace(label_dict)
        y2 = y2.values.astype(np.float64)

        dc = DataContainer(X_train=X1, X_test=X2, y_train=y1, y_test=y2)
        pickle.dump(dc, open(cache_filename, "wb"))

    else:
        dc = pickle.load(open(cache_filename, "rb"))

    
    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")

    return dc

def load_ICB(train_matrices=config.ICB_TRAIN_MAT, 
            train_labels=config.ICB_TRAIN_CLS, 
            test_matrix=config.ICB_TEST_MAT, 
            test_label=config.ICB_TEST_CLS,
            t_cell_signatures=config.ICB_TCELL_SIG) -> DataContainer:
    
    # Time stamp
    start_time = time.time()

    cache_filename = os.path.join(os.getcwd(), 'data', '+'.join(train_matrices) + "-" + test_matrix + ".pkl")

    # If not cached, cache data
    if not os.path.exists(cache_filename):
        print('caching...')

        # Read training data
        train_mat = []
        train_cls = []
        
        # Read matrix files
        for filename in train_matrices:
            filepath = os.path.join(os.getcwd(), 'data', filename)
            if os.path.isfile(filepath):
                mat = pd.read_csv(filepath, sep=',', index_col=0)
                # Remove duplicated genes
                mat = mat[~mat.index.duplicated(keep='first')]
                train_mat.append(mat)
                
            else:
                print(f"FileNotFoundError: {filepath} does not exist")
                exit()

        # Read label files
        for filename in train_labels:
            filepath = os.path.join(os.getcwd(), 'data', filename)
            if os.path.isfile(filepath):
                train_cls.append(pd.read_csv(filepath, sep=',', index_col=0, header=None))
            else:
                print(f"FileNotFoundError: {filepath} does not exist")
                exit()

        # Read test data
        filepath = os.path.join(os.getcwd(), 'data', test_matrix)
        if os.path.isfile(filepath):
            test_mat = pd.read_csv(filepath, sep=',', index_col=0)
            test_mat = test_mat[~test_mat.index.duplicated(keep='first')]
        else:
            print(f"FileNotFoundError: {filepath} does not exist")
            exit()

        filepath = os.path.join(os.getcwd(), 'data', test_label)
        if os.path.isfile(filepath):
            test_cls = pd.read_csv(filepath, sep=',', index_col=0, header=None)
        else:
            print(f"FileNotFoundError: {filepath} does not exist")
            exit()

        # Get overlapping gene index
        features = train_mat[0].index
        for i in range(1, len(train_mat)):
            features = features.intersection(train_mat[i].index)

        features = features.intersection(test_mat.index).unique()

        # Concatenate training data and leave only overlapping features
        train_mat = pd.concat([x.loc[features] for x in train_mat], axis=1)
        train_cls = pd.concat([y for y in train_cls], axis=0)

        # Leave only overlapping features out in test data
        test_mat = test_mat.loc[features]

        # Covert FPKM into TPM
        train_mat = train_mat / train_mat.sum() * 1000000
        test_mat = test_mat / test_mat.sum() * 1000000

        # log2(x+1) transform
        train_mat = np.log2(train_mat + 1)
        test_mat = np.log2(test_mat + 1)

        # Read T cell signature file
        filepath = os.path.join(os.getcwd(), 'data', t_cell_signatures)
        if os.path.isfile(filepath):
            t_cell_sig = pd.read_csv(filepath, sep=',', index_col=0, header=None)
        else:
            print(f"FileNotFoundError: {filepath} does not exist")
            exit()

        # Leave only features in T cell signatures
        features = features.intersection(t_cell_sig.index).unique()
        train_mat = train_mat.loc[features]
        test_mat = test_mat.loc[features]

        # Transpose matrix and transform pandas dataframe to numpy array
        X_train = train_mat.T.values.astype(np.float)
        y_train = train_cls.values.astype(np.int).flatten()
        X_test = test_mat.T.values.astype(np.float)
        y_test = test_cls.values.astype(np.int).flatten()
        
        dc = DataContainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        pickle.dump(dc, open(cache_filename, "wb"))

    else:
        dc = pickle.load(open(cache_filename, "rb"))

    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")

    return dc