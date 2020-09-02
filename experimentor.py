import os
import time
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

class DataContainer(object):
    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None):
        # Training and test data holders
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

class Experimentor(object):
    def __init__(self, data : DataContainer, exp_name : str):
        # Data name
        self.exp_name = exp_name

        # Create directory with experiment name
        self.result_path = os.path.join(os.getcwd(), 'results', exp_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Training and test data holders
        self.X_train = data.X_train
        self.X_test = data.X_test
        self.y_train = data.y_train
        self.y_test = data.y_test

        # Augmentation holders
        self.aug_name = None
        self.aug_rates = None
        self.X_augs = None
        self.y_augs = None
        self.X_train_augs = None
        self.y_train_augs = None

        # Classifiers
        self.classifiers = [SVC(probability=True, random_state=0, gamma='scale'), RandomForestClassifier(random_state=0, n_estimators=100), MLPClassifier(random_state=0, hidden_layer_sizes=(128, 64, 32), max_iter=500)]
        self.classifier_names = ["SVM", "RF", "NN"]

        # Standardization
        self._standardize()

        # Feature Selection
        self._select_feature()

        # Distribution
        #self._draw_histogram(Xs=[self.X_train, self.X_test], Xs_labels=["X_train", "X_test"])

        # Viz
        #self.visualize(X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test)

    def _standardize(self):
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def _select_feature(self, n_estimators=500, max_features=256):
        clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion="entropy", random_state=0)
        clf = clf.fit(self.X_train, self.y_train)
        #print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True, max_features=max_features)
        self.X_train = model.transform(self.X_train)
        self.X_test = model.transform(self.X_test)

    def _draw_histogram(self, Xs : list, Xs_labels : list):
        num_Xs = len(Xs)
        fig, axs = plt.subplots(num_Xs, figsize=(15, 4 * num_Xs))
        for i in range(num_Xs):
            axs[i].hist(Xs[i].flatten(), bins='auto')
            axs[i].text(x=0.02, y=0.9, s=Xs_labels[i], transform=axs[i].transAxes)
        plt.show()

    def visualize(self, X_train, y_train, X_test, y_test, lower_bound=-5, upper_bound=5):
        # Sort data by class label
        idx = np.argsort(y_train)
        X_train_sorted = X_train[idx]
        y_train_sorted = y_train[idx]
        idx = np.argsort(y_test)
        X_test_sorted = X_test[idx]
        y_test_sorted = y_test[idx]

        # Get the number of unique class labels and the number of plots
        classes = np.unique(y_train)
        num_class = len(classes)
        num_plots = num_class * 2

        fig, axs = plt.subplots(num_plots, figsize=(15,4*num_plots))
        
        for i in range(0, num_plots):
            if i < (num_plots / 2):
                # Training data viz
                ms = axs[i].matshow(X_train_sorted[y_train_sorted == classes[i]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)
            else:
                # Test data viz
                ms = axs[i].matshow(X_test_sorted[y_test_sorted == classes[i - num_class]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)

        # Location and size of colorbar: [coordinate1, coordinate2 inthe figure, colorbar width, height]
        cax = fig.add_axes([0.94, 0.2, 0.02, 0.4])
        
        # Add colorbar
        fig.colorbar(mappable=ms, cax=cax, extend='both')

        # Show figure
        plt.show()

    def classify_without_augmentation(self):
        with open(os.path.join(self.result_path, 'noAug.txt'), "w") as f:
            # Write result header
            f.write("Clf\tAUROC\tAUPRC\tACC  \tREC  \tPRE  \tF1  \n")

            for clf, clf_name in zip(self.classifiers, self.classifier_names):
                clf.fit(self.X_train, self.y_train)
                y_pred = clf.predict(self.X_test)
                y_prob = clf.predict_proba(self.X_test)

                precisions, recalls, _ = precision_recall_curve(self.y_test, y_prob[:, 1])

                # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                auroc = round(roc_auc_score(self.y_test, y_prob[:, 1]), 3)
                auprc = round(auc(recalls, precisions), 3)
                acc = round(accuracy_score(self.y_test, y_pred), 3)
                rec = round(recall_score(self.y_test, y_pred), 3)
                pre = round(precision_score(self.y_test, y_pred), 3)
                f1 = round(f1_score(self.y_test, y_pred), 3)

                f.write(f"{clf_name}\t{auroc}\t{auprc}\t{acc}\t{rec}\t{pre}\t{f1}\n")

    def classify_with_augmentation(self):
         # Time stamp
        start_time = time.time()

        with open(os.path.join(self.result_path, self.aug_name + 'Aug.txt'), "w") as f:
            # Write result header
            f.write("Clf\tAugRate\tAUROC\tAUPRC\tACC  \tREC  \tPRE  \tF1  \n")

            for clf, clf_name in zip(self.classifiers, self.classifier_names):
                for i in range(len(self.aug_rates)):
                    clf.fit(self.X_train_augs[i], self.y_train_augs[i])
                    y_pred = clf.predict(self.X_test)
                    y_prob = clf.predict_proba(self.X_test)

                    precisions, recalls, _ = precision_recall_curve(self.y_test, y_prob[:, 1])

                    # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                    auroc = round(roc_auc_score(self.y_test, y_prob[:, 1]), 3)
                    auprc = round(auc(recalls, precisions), 3)
                    acc = round(accuracy_score(self.y_test, y_pred), 3)
                    rec = round(recall_score(self.y_test, y_pred), 3)
                    pre = round(precision_score(self.y_test, y_pred), 3)
                    f1 = round(f1_score(self.y_test, y_pred), 3)

                    f.write(f"{clf_name}\t{self.aug_rates[i]}\t{auroc}\t{auprc}\t{acc}\t{rec}\t{pre}\t{f1}\n")
        
        print(f"--- Classified with {self.aug_name} augmentation in {round(time.time() - start_time, 2)} seconds ---")



            

        

    
