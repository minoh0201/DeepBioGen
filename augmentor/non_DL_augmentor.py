# Fix numpy random seed
from numpy.random import seed
seed(0)
# from tensorflow import set_random_seed
# set_random_seed(2)

import time
import numpy as np
from experimentor import Experimentor
from sklearn import mixture
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

def random(exp : Experimentor, aug_rates : list):
    # Time stamp
    start_time = time.time() 

    # Set augmentation name
    exp.aug_name = "random"

    # Store aug rates
    exp.aug_rates = aug_rates

    # The largest number of augmented samples
    max_aug_samples = exp.X_train.shape[0] * aug_rates[-1]

    # Statistics of training distribution
    mean = np.mean(exp.X_train, axis=0)
    std = np.std(exp.X_train, axis=0)

    # Generate random samples from training distribution
    X_fake = np.random.normal(mean, std, (max_aug_samples, exp.X_train.shape[1]))

    # Generate random class labels
    y_fake = np.random.randint(len(np.unique(exp.y_train)), size=max_aug_samples)

    # Store augmentations
    exp.X_augs = []
    exp.y_augs = []
    exp.X_train_augs = []
    exp.y_train_augs = []
    for aug_rate in aug_rates:
        num_aug_samples = int(exp.X_train.shape[0] * aug_rate)

        # Augmentation data alone
        exp.X_augs.append(X_fake[:num_aug_samples])
        exp.y_augs.append(y_fake[:num_aug_samples])

        # Training data + Augmentation data
        exp.X_train_augs.append(np.concatenate((exp.X_train, X_fake[:num_aug_samples])))
        exp.y_train_augs.append(np.concatenate((exp.y_train, y_fake[:num_aug_samples])))

    print(f"--- Augmented with {exp.aug_name} in {round(time.time() - start_time, 2)} seconds ---")

def gmm(exp : Experimentor, aug_rates : list):
    # Time stamp
    start_time = time.time() 

    # Set augmentation name
    exp.aug_name = "gmm"

    # The number of components in GMM models
    max_gmm_components = 10

    # Store aug rates
    exp.aug_rates = aug_rates

    # Class labels
    class_labels = np.unique(exp.y_train)

    # The largest number of augmented samples
    max_aug_samples = exp.X_train.shape[0] * aug_rates[-1]
    max_aug_samples_for_each_class = int(max_aug_samples/len(class_labels))

    # GMM augmentation
    gmm_BICs = []
    for label in class_labels:
        gmm_BICs.append(np.full(max_gmm_components + 1, np.inf))

    # Select the best GMM models by varying k components
    for i in range(len(class_labels)):
        for n_comp in range(2, max_gmm_components + 1):
            gmm = mixture.GaussianMixture(n_components = n_comp, random_state=0)
            gmm.fit(exp.X_train[exp.y_train==class_labels[i], :])
            gmm_BICs[i][n_comp] = gmm.bic(exp.X_train[exp.y_train==class_labels[i], :])

        print(print(f'gmm_BICs of class {class_labels[i]}:{gmm_BICs[i]}, The best # of components: {np.argmin(gmm_BICs[i])}'))

    # Augment data using the best model identified by BIC score
    gmm_samples = []
    gmm_sample_classes = []
    for i in range(len(class_labels)):
        gmm = mixture.GaussianMixture(n_components = np.argmin(gmm_BICs[i]), random_state=0, verbose=2)
        gmm.fit(exp.X_train[exp.y_train==class_labels[i], :])
        # Sampling
        gmm_sample, _ = gmm.sample(max_aug_samples_for_each_class)
        # Append samples and corresponding classes
        gmm_samples.append(gmm_sample)
        gmm_sample_classes.append(np.full(max_aug_samples_for_each_class, class_labels[i]).astype('int64'))

    
    # Concatenate samples of each class
    X_fake = np.concatenate(gmm_samples)
    y_fake = np.concatenate(gmm_sample_classes)
    X_fake, y_fake = shuffle(X_fake, y_fake, random_state=0)

    # Store augmentations
    exp.X_augs = []
    exp.y_augs = []
    exp.X_train_augs = []
    exp.y_train_augs = []
    for aug_rate in aug_rates:
        num_aug_samples = int(exp.X_train.shape[0] * aug_rate)

        # Augmentation data alone
        exp.X_augs.append(X_fake[:num_aug_samples])
        exp.y_augs.append(y_fake[:num_aug_samples])

        # Training data + Augmentation data
        exp.X_train_augs.append(np.concatenate((exp.X_train, X_fake[:num_aug_samples])))
        exp.y_train_augs.append(np.concatenate((exp.y_train, y_fake[:num_aug_samples])))

    print(f"--- Augmented with {exp.aug_name} in {round(time.time() - start_time, 2)} seconds ---")
        
def smote(exp : Experimentor, aug_rates : list):
    # Time stamp
    start_time = time.time() 

    # Set augmentation name
    exp.aug_name = "smote"

    # Store aug rates
    exp.aug_rates = aug_rates

    # Class labels
    class_labels = np.unique(exp.y_train)

    # The largest number of augmented samples
    max_aug_samples = exp.X_train.shape[0] * aug_rates[-1]
    max_aug_samples_for_each_class = int(max_aug_samples/len(class_labels))

    # Augment data using SMOTE
    sampling_strategy = dict()
    for label in class_labels:
        sampling_strategy[label] = max_aug_samples_for_each_class
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=0)
    X_smote, y_smote = smote.fit_resample(exp.X_train, exp.y_train)
    # Take augmentations only by dropping training data in the output of smote
    num_train_data = exp.X_train.shape[0]
    X_fake, y_fake = X_smote[num_train_data:], y_smote[num_train_data:]
    X_fake, y_fake = shuffle(X_fake, y_fake, random_state=0)

    # Store augmentations
    exp.X_augs = []
    exp.y_augs = []
    exp.X_train_augs = []
    exp.y_train_augs = []
    for aug_rate in aug_rates:
        num_aug_samples = int(exp.X_train.shape[0] * aug_rate)

        # Augmentation data alone
        exp.X_augs.append(X_fake[:num_aug_samples])
        exp.y_augs.append(y_fake[:num_aug_samples])

        # Training data + Augmentation data
        exp.X_train_augs.append(np.concatenate((exp.X_train, X_fake[:num_aug_samples])))
        exp.y_train_augs.append(np.concatenate((exp.y_train, y_fake[:num_aug_samples])))

    print(f"--- Augmented with {exp.aug_name} in {round(time.time() - start_time, 2)} seconds ---")

        