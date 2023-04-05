```
Environment setup:
conda create --name medical python=3.8
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install sk-video matplotlib gdown opencv-python scikit-learn scikit-image nibabel
pip install h5py tensorboardX MedPy

```

Dataset Stats:

- BraTS2019
	- Train/Val/test Images - 250/35/60

Evaluation Metrics:
- Dice SCore, Jaccard, ASD, HD95
