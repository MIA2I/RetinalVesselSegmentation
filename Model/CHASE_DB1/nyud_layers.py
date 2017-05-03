import sys
sys.path.append('/home/john/caffe-master/python/')

import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

class NYUDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from NYUDv2
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 40 class task defined by

        S. Gupta, R. Girshick, p. Arbelaez, and J. Malik. Learning rich features
        from RGB-D images for object detection and segmentation. ECCV 2014.

    with 0 as the void label and 1-40 the classes.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - nyud_dir: path to NYUDv2 dir
        - split: train / val / test
        - tops: list of tops to output from {color, depth, hha, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for NYUDv2 semantic segmentation.

        example: params = dict(nyud_dir="/path/to/NYUDVOC2011", split="val",
                               tops=['color', 'hha', 'label'])
        """
        # config
        params = eval(self.param_str)
        self.nyud_dir = params['nyud_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # store top data for reshape + forward
        self.data = {}

        # means
        self.mean_bgr = np.array((126.8371, 69.0155, 41.4216), dtype=np.float32)
        self.mean_d = np.array((44.3911), dtype=np.float32)

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.nyud_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False
        self.random = False
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.indices[self.idx])
            top[i].reshape(1, *self.data[t].shape)

    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx):
        if top == 'color':
            return self.load_image(idx)
        elif top == 'label':
            return self.load_label(idx)
        elif top == 'RGB':
            return self.load_color(idx)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_color(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/Image/{}.png'.format(self.nyud_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))

        return in_

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/Image/{}.png'.format(self.nyud_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        #in_ = in_[:,:,::-1]
        in_ -= self.mean_d
        #in_ = in_.transpose((2,0,1))
        in_ = in_[np.newaxis, ...]

        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        label = scipy.io.loadmat('{}/Mat/{}.mat'.format(self.nyud_dir, idx))['label'].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label
