import data_loader
from experimentor import DataContainer
from experimentor import Experimentor
from augmentor import non_DL_augmentor
from augmentor import wGAN_augmentor

if __name__ == "__main__":
    # T2D to WT2D
    t2d_data = data_loader.load_T2D()

    exp = Experimentor(data=t2d_data, exp_name="T2D_WT2D")

    #non_DL_augmentor.random(exp = exp, aug_rates=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #non_DL_augmentor.gmm(exp = exp, aug_rates=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #non_DL_augmentor.smote(exp = exp, aug_rates=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    wGAN_augmentor.deepbiogen(exp = exp, aug_rates=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], num_clusters=2, max_gans=3, num_epochs=100, batch_size=128, sample_interval=50)

    exp.classify_with_wGAN_augmentation()

    exp.classify_without_augmentation()

    exp.classify_with_augmentation()

    exp.visualize(X_train=exp.X_train, y_train=exp.y_train, X_test=exp.X_test, y_test=exp.y_test)
    exp.visualize(X_train=exp.X_train, y_train=exp.y_train, X_test=exp.X_augs[1], y_test=exp.y_augs[1])

    

    print("end")
