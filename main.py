import data_loader
from experimentor import DataContainer
from experimentor import Experimentor
from augmentor import non_DL_augmentor
from augmentor import wGAN_augmentor

def run_exps(data : DataContainer, exp_name : str, aug_rates: list):

    # Baseline (no augmentation)
    exp = Experimentor(data=data, exp_name=exp_name)
    exp.classify_without_augmentation()

    # Non-DL augmentations and the subsequent classifications and visualizations
    non_DL_augmentors = [non_DL_augmentor.random, non_DL_augmentor.gmm, non_DL_augmentor.smote]

    for augmentor in non_DL_augmentors:
        exp = Experimentor(data=data, exp_name=exp_name)
        augmentor(exp = exp, aug_rates=aug_rates)
        exp.classify_with_non_DL_augmentation()
        exp.visualize_aug(X_train=exp.X_train, y_train=exp.y_train, X_test=exp.X_test, y_test=exp.y_test, X_aug=exp.X_augs[1], y_aug=exp.y_augs[1])
        exp.draw_histogram(Xs=[exp.X_train, exp.X_test, exp.X_augs[1]], Xs_labels=["X_train", "X_test", "X_aug"])

    # wGAN augmentation
    exp = Experimentor(data=data, exp_name=exp_name)
    wGAN_augmentor.deepbiogen(exp = exp, aug_rates=aug_rates, num_clusters=2, max_gans=10, num_epochs=6000, batch_size=128, sample_interval=2000)
    exp.classify_with_wGAN_augmentation()
    exp.visualize_aug(X_train=exp.X_train, y_train=exp.y_train, X_test=exp.X_test, y_test=exp.y_test, X_aug=exp.X_augs[5][1], y_aug=exp.y_augs[5][1])
    exp.draw_histogram(Xs=[exp.X_train, exp.X_test, exp.X_augs[5][1]], Xs_labels=["X_train", "X_test", "X_aug"])

if __name__ == "__main__":

    # Set augmentation rates to retrieve
    aug_rates = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # T2D to WT2D
    t2d_data = data_loader.load_T2D()
    run_exps(data=t2d_data, exp_name="T2D_WT2D", aug_rates=aug_rates)

    print("End")
