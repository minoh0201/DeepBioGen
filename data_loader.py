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