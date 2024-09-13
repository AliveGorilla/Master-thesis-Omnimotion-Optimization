Proposal model optimisation to the "Tracking Everything Everywhere All at Once" paper

Here's the original installation: 

#### [Project Page](https://omnimotion.github.io/) | [Paper](https://arxiv.org/pdf/2306.05422.pdf) | [Video](https://www.youtube.com/watch?v=KHoAG3gA024)
## Installation
The code is tested with `python=3.8` and `torch=1.10.0+cu111` on an A100 GPU.
```
git clone --recurse-submodules https://github.com/qianqianwang68/omnimotion/
cd omnimotion/
conda create -n omnimotion python=3.8
conda activate omnimotion
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tensorboard scipy opencv-python tqdm tensorboardX configargparse ipdb kornia imageio[ffmpeg]
```

## Training
1. Please refer to the [preprocessing instructions](preprocessing/README.md) for preparing input data 
   for training OmniMotion. We also provide some processed [data](https://omnimotion.cs.cornell.edu/dataset/)
   that you can download, unzip and directly train on. (Note that depending on the network speed, 
   it may be faster to run the processing script locally than downloading the processed data).
   
2.  With processed input data, run the following command to start training:
    ```
    python train.py --config configs/default.txt --data_dir {sequence_directory}
    ```


## Citation to original model
```
@article{wang2023omnimotion,
    title   = {Tracking Everything Everywhere All at Once},
    author  = {Wang, Qianqian and Chang, Yen-Yu and Cai, Ruojin and Li, Zhengqi and Hariharan, Bharath and Holynski, Aleksander and Snavely, Noah},
    journal = {ICCV},
    year    = {2023}
}
```
