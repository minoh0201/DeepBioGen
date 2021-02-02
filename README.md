# DeepBioGen

A sequencing data augmentation procedure that establishes visual patterns from sequencing profiles and that generates realistic sequencing profiles based on conditional Wasserstein GAN capturing the visual patterns

## Starting guide
Require GPU machine, git, Anaconda3, docker, docker image of `tensorflow/tensorflow:1.13.2-gpu-py3-jupyter`

1. Run docker image equipped with tensorflow:1.13.2-gpu
```
docker run -it --rm -p 8888:8888 -v ~/[DEEPBIOGEN DIR NAME]:/tf/[DEEPBIOGEN DIR NAME] --runtime=nvidia tensorflow/tensorflow:1.13.2-gpu-py3-jupyter bash
```

1. Clone the repo into your local directory
```
git clone [REPO URL]
```
2. Create a virtual environment
```
conda create -n deepbiogen python=3.6
```
3. Activate the virtual environment
```
conda activate deepbiogen
```
4. Install required packages
```
pip install -r requirements.txt
```
5. Run DeepBioGen and check usage
```
python main.py -h
```

## Experiment guide

1. Estimate # of visual clusters and # of GANs with the elbow method (e.g. ICB data - RNA-seq tumor expression profiles with anti-PD1 therapy response labels).
```
python main.py -d ICB --elbow
```

2. Move to results directory `../DeepBioGen/results/ICB`, and check `featurewise_WSS.png` and `samplewise_WSS.png` files to estimate # of visual clusters and # of GANs, respectively.

3. Run experiment with the estimated parameters (e.g. # of visual clusters: 4; # of GANs: 5).
```
python main.py -d ICB --num_clusters 4 --num_gans 5
```
