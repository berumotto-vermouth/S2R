import numpy as np
import scipy
from torch.utils.data import Dataset
from imresize import imresize
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation, rgb2gray
import torch
import torchvision
from scipy import signal, ndimage


# np.random.seed(0)
# torch.manual_seed(0)


def my_prob_map(image):
    g_im = rgb2gray(image)
    sx = ndimage.sobel(g_im, axis=0)
    sy = ndimage.sobel(g_im, axis=1)
    sobel = np.hypot(sx, sy)
    sobel *= 1 / np.max(sobel)  # возможно плохо
    sobel = np.clip((sobel - 0.7), 0, 1)
    sobel *= 1 / np.max(sobel)

    p_map = signal.convolve2d(sobel, np.ones((5, 5)) / 25, mode='same')
    p_map /= np.sum(p_map)

    return p_map.flatten()


class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    def __init__(self, conf, gan):
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # Read input image
        self.input_image = read_image(conf.input_image_path) / 255.
        self.input_lr = self.input_image if not conf.weakly_supervised_path else read_image(
            conf.weakly_supervised_path) / 255.
        print(conf.weakly_supervised_path)
        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real_image)
        self.input_image_for_crop = np.copy(self.input_image)
        self.input_lr_for_crop = np.copy(self.input_lr)
        print(self.input_image.shape, self.input_lr.shape)

        # self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # Create prob map for choosing the crop
        # print(len(self.input_image) * len(self.input_image[0]), my_prob_map(self.input_image).shape)
        self.crop_indices_for_g = np.random.choice(a=(len(self.input_image) * len(self.input_image[0])),
                                                   size=conf.max_iters, p=my_prob_map(self.input_image))
        self.crop_indices_for_d = np.random.choice(a=(len(self.input_lr) * len(self.input_lr[0])), size=conf.max_iters,
                                                   p=my_prob_map(self.input_lr))
        # self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        g_in = self.next_crop(for_g=True, idx=idx)  # comment idx
        d_in = self.next_crop(for_g=False, idx=idx)  # comment idx

        return g_in, d_in

    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        image = self.input_image_for_crop if for_g else self.input_lr_for_crop
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)

        
        #######################################
        if left < 0:
            left = 0

        #######################################



        crop_im = np.copy(image[top:top + size, left:left + size, :])
        # if not for_g:  # Add noise to the image for d
        # crop_im += np.random.randn(*crop_im.shape) / 255.0
        return im2tensor(crop_im)

    def make_list_of_crop_indices(self, conf):
        iterations = conf.max_iters
        prob_map_big, prob_map_sml = self.create_prob_maps(scale_factor=conf.scale_factor)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps(self, scale_factor):
        # Create loss maps for input image and downscaled one
        loss_map_big = create_gradient_map(self.input_image)
        loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'))
        # Create corresponding probability maps
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
            self.input_lr = self.input_lr[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

        shape_lr = self.input_lr.shape
        self.input_lr = self.input_lr[:-(shape_lr[0] % sf), :, :] if shape_lr[0] % sf > 0 else self.input_lr
        self.input_lr = self.input_lr[:, :-(shape_lr[1] % sf), :] if shape_lr[1] % sf > 0 else self.input_lr

    def get_top_left(self, size, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        image = self.input_image if for_g else self.input_lr
        row, col = int(center / image.shape[1]), center % image.shape[1]
        top, left = min(max(0, row - size // 2), image.shape[0] - size), min(max(0, col - size // 2),
                                                                             image.shape[1] - size)
        # Choose even indices (to avoid misalignment with the loss map for_g)
        return top - top % 2, left - left % 2

    def my_next_crop(self, for_g):
        size_of_crop = self.g_input_shape if for_g else self.d_input_shape
        cropped_image = torchvision.transforms.RandomCrop(size_of_crop)(im2tensor(self.input_image))

        if not for_g:
            cropped_image += im2tensor(np.random.randn(size_of_crop, size_of_crop, 3) / 255.)

        return cropped_image
