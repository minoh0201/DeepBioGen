import argparse

import os
import data_loader
import config
from experimentor import DataContainer
from experimentor import Experimentor
from augmentor import non_DL_augmentor
from augmentor import wGAN_augmentor

from sklearn.model_selection import KFold

def run_exps(data : DataContainer, exp_name : str, aug_rates: list, num_clusters=None, num_gans=None):

    # Baseline (no augmentation)
    exp = Experimentor(data=data, exp_name=exp_name)
    exp.classify_without_augmentation()

    # Non-DL augmentations and the subsequent classifications and visualizations
    non_DL_augmentors = [non_DL_augmentor.random, non_DL_augmentor.gmm, non_DL_augmentor.smote]

    for augmentor in non_DL_augmentors:
        exp = Experimentor(data=data, exp_name=exp_name)
        augmentor(exp = exp, aug_rates=aug_rates, save_all_data=True)
        exp.classify_with_non_DL_augmentation()
        exp.visualize_aug(X_train=exp.X_train, y_train=exp.y_train, X_test=exp.X_test, y_test=exp.y_test, X_aug=exp.X_augs[1], y_aug=exp.y_augs[1])
        exp.draw_histogram(Xs=[exp.X_train, exp.X_test, exp.X_augs[1]], Xs_labels=["X_train", "X_test", "X_aug"])

    # wGAN augmentation
    if num_clusters == None:
        c_range = range(1, 11)
    else:
        c_range = range(num_clusters, num_clusters+1)
    
    if num_gans == None:
        max_gans = 10
    else:
        max_gans = num_gans

    for i in c_range:  # number of data clusters
        exp = Experimentor(data=data, exp_name=exp_name)
        wGAN_augmentor.deepbiogen(exp = exp, 
                                aug_rates=aug_rates, 
                                num_clusters=i, 
                                max_gans=max_gans, 
                                num_epochs=6000, 
                                batch_size=128, 
                                sample_interval=2000,
                                save_all_data=True)
        exp.classify_with_wGAN_augmentation(fixed_num_gans=num_gans)

    del exp
    del data

def run_cv_exps(data : DataContainer, exp_name : str, aug_rates: list, num_clusters: int, num_gans: int):
    # Cross validation within training data
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    kf.get_n_splits(data.X_train)

    current_fold = 0
    for train_index, test_index in kf.split(data.X_train):
        current_fold += 1
        current_exp_name = exp_name + f'_{current_fold}'
        print(current_exp_name)
        X_train_folds, y_train_folds = data.X_train[train_index], data.y_train[train_index]
        X_test_fold, y_test_fold = data.X_train[test_index], data.y_train[test_index]

        run_exps(data=DataContainer(X_train=X_train_folds, X_test=X_test_fold, y_train=y_train_folds, y_test=y_test_fold), 
                exp_name = current_exp_name,
                aug_rates=aug_rates,
                num_clusters=num_clusters,
                num_gans=num_gans)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="dataset indicator (e.g. T2D_WT2D or ICB)", type=str, choices=["T2D_WT2D", "ICB", "ICB_COMB"])
    parser.add_argument("--elbow", help="Run elbow method only and quit", action='store_true')
    parser.add_argument("--cv", help="cross validation on training data only", action='store_true')
    parser.add_argument("-r", "--aug_rates", help="augmentation rates (e.g. 0.5,1,2,4)", type=str, default="0.5, 1, 2, 4, 8, 16, 32")
    parser.add_argument("--num_clusters", help="The number of visual clusters", type=int, default=None)
    parser.add_argument("--num_gans", help="The number of CWGANs", type=int, default=None)

    args = parser.parse_args()
    print(args)

    # Set augmentation rates to retrieve
    aug_rates = [float(x) for x in args.aug_rates.split(',')]
 
    # Load data according to argument
    if args.data == 'ICB':
        data = data_loader.load_ICB()
    elif args.data == 'ICB_COMB':
        # ICB Combination
        data = data_loader.load_ICB(train_matrices=config.ICB_TRAIN_MAT, train_labels=config.ICB_TRAIN_CLS, test_matrix=config.ICB_COMB_TEST_MAT, test_label=config.ICB_COMB_TEST_CLS, t_cell_signatures=config.ICB_TCELL_SIG)
    elif args.data == 'T2D_WT2D':
        data = data_loader.load_T2D()

    # Run elbow only
    if args.elbow:
        exp = Experimentor(data=data, exp_name=args.data)
        exit()

    # Run experiments
    if args.cv:
        run_cv_exps(data=data, exp_name=args.data, aug_rates=aug_rates, num_clusters=args.num_clusters, num_gans=args.num_gans)
    else:
        if args.num_clusters != None and args.num_gans != None:
            run_exps(data=data, exp_name=args.data, aug_rates=aug_rates, num_clusters=args.num_clusters, num_gans=args.num_gans)
        else:
            run_exps(data=data, exp_name=args.data, aug_rates=aug_rates)

    print("End")
