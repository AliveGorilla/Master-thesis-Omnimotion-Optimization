# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

import cv2
import torch

#from utils import utils
#from models import vision_transformer as vits

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device = torch.device("cuda")
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    # Set the device to CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")

def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    print('extracting feature for one frame')
    #print('*' * 50)
    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    #print('out model:', out)
    #print('out model shape:', out.shape)
    #out = out[:, 1:, :]  # we discard the [CLS] token
    '''
    print('out model shape:', out.shape)
    print('out: ', out)
    print('patch_embed: ', model.patch_embed)
    print('patch_embed.patch_size: ', model.patch_embed.patch_size)
    print('frame.shape: ', frame.shape)
    print('frame.shape[1]: ', frame.shape[1])
    print('frame.shape[2]: ', frame.shape[2])
    '''

    h, w = int(frame.shape[1] / model.patch_embed.patch_size[0]), int(frame.shape[2] / model.patch_embed.patch_size[0])
    #print('h, w: ', h, w)
    dim = out.shape[-1]
    '''
    print('dim: ', dim)
    print('out before reshape: ', out)
    print('out.shape before reshape: ', out.shape)
    print('out[0].shape before reshape: ', out[0].shape)
    '''
    out = out[0].reshape(h, w, dim)
    #print('out.shape after reshape: ', out.shape)
    #print('out after reshape: ', out)
    out = out.reshape(-1, dim)
    #print('out: ', out)
    if return_h_w:
        return out, h, w
    return out


def read_frame(frame_dir, scale_size=[420]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if (ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 56) * 56)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 56) * 56)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--pretrained_weights', default='.',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=14, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default=".", help='Path where to save segmentations')
    parser.add_argument('--data_path', default='/path/to/davis/', type=str)
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()

    #print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # building network
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model = model.cuda()
    #print('model.patch_embed.patch_size: ', model.patch_embed.patch_size)
    #utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key)
    # OLD (model, pretrained_weights, checkpoint_key, model_name, patch_size)
    # (model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size) 
    # NEW (model, pretrained_weights, checkpoint_key)
    # (model, args.pretrained_weights, args.checkpoint_key)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    scene_dir = args.data_dir

    frame_list = sorted(glob.glob(os.path.join(scene_dir, 'color', '*')))
    save_dir = os.path.join(scene_dir, 'features', 'dino')

    print('computing dino features for {}...'.format(scene_dir))
    os.makedirs(save_dir, exist_ok=True)

    for frame_path in tqdm(frame_list):
        frame, ori_h, ori_w = read_frame(frame_path)
        '''
        print('frame: ', frame)
        print('frame.shape[1]: ', frame.shape[1])
        print('frame.shape[2]: ', frame.shape[2])
        print('ori_h: ', ori_h)
        print('ori_w: ', ori_w)
        '''
        frame_feat, h, w = extract_feature(model, frame, return_h_w=True)  # dim x h*w
        frame_feat = frame_feat.reshape(h, w, -1)
        frame_feat = frame_feat.cpu().numpy()
        frame_name = os.path.basename(frame_path)
        np.save(os.path.join(save_dir, frame_name + '.npy'), frame_feat)

    print('computing dino features for {} is done \n'.format(scene_dir))
