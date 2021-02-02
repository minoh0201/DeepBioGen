# DeepBioGen

A sequencing data augmentation procedure that establishes visual patterns from sequencing profiles and that generates realistic sequencing profiles based on conditional Wasserstein GAN capturing the visual patterns

## Starting guide
Require GPU machine, git, Anaconda3, docker, docker image of "tensorflow/tensorflow:1.13.2-gpu-py3-jupyter"

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
5. Run DeepBioGen
```
python main.py
```
