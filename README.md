Proposal model optimisation to the "Tracking Everything Everywhere All at Once" paper

Links to the original paper:
#### [Project Page](https://omnimotion.github.io/) | [Paper](https://arxiv.org/pdf/2306.05422.pdf) | [Video](https://www.youtube.com/watch?v=KHoAG3gA024)

## Here's the original installation: 
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
## Modification

A more detailed description of this work can be found in my thesis. Below is a summary of the modifications and changes I made:

1. The original implementation consisted of a large number of different loss functions. I simplified the model by removing unnecessary losses (specifically, the "smoothness loss") after conducting an empirical analysis.

2. I modified the hard-mining method. Instead of randomly selecting regions with the highest probability of error, I aggregated these maps over the course of training. This allowed the model to focus on regions with consistently high errors over time.

3. The original OmniMotion model did not include 3D coordinate encoding. I added a simple encoding option by applying periodic functions to the 3-dimensional input coordinates, producing a 66-dimensional vector as output.

4. During the training stage, the Invertible Neural Network trained the time representation (frame number) more than once in each epoch. I addressed this by locking the training of the time representation at the 40,000-epoch threshold and using the final result as a global representation ("freezing" it).

5. In addition to freezing the number of epochs, I applied a faster and more advanced method, [TCNN](https://github.com/NVlabs/tiny-cuda-nn), to further enhance the training process.

## Future Ideas

During my research, I identified several advanced ideas and solutions that could further improve the project:

1. The [CoTracker](https://github.com/facebookresearch/co-tracker) model demonstrates better and more accurate results compared to RAFT. Through testing, I confirmed that RAFT can be replaced. However, I encountered optimization issues that I hope to address in the future.

2. In the pre-processing stage, the DINO model used for appearance checking can be upgraded to the more recent DINO v2 for better performance.

3. Several improvements to OmniMotion suggest that incorporating additional information (e.g., depth maps) can significantly enhance the overall quality of the results.

While these ideas remain unimplemented due to time constraints, I plan to explore them further as a future PhD student or as part of a personal project.
