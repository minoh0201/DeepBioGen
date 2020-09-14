# DeepBioGen

Require GPU machine, Anaconda3, docker, docker image of "tensorflow/tensorflow:1.13.2-gpu-py3-jupyter"

1. git clone
2. conda create -n deepbiogen python=3.6
3. conda activate deepbiogen
4. pip install -r requirements.txt
5. python main.py



docker run -it --rm -p 8888:8888 -v ~/DeepBioGen:/tf/DeepBioGen --runtime=nvidia tensorflow/tensorflow:1.13.2-gpu-py3-jupyter bash
