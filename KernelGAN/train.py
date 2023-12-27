import os
from time import sleep

import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner


'''
writer = SummaryWriter()
r = 5
for i in range(100):
    sleep(0.5)
    writer.add_scalars('run_14h', {'xsinx':(i + 1) * np.sin(i/r),
                                    'xcosx':(i + 1) * np.cos(i/r),
                                    'tanx': np.tan(i/r)}, 3 * i)
writer.close()
'''


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    gan.finish()


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--ground-dir', '-t', type=str, default='ground_truth_images', help='path to image ground-truth directory.')


    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    prog.add_argument("--ws_dir", type=str, help="path for ws_lr images")
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              
              '--ground_image_path', os.path.join(args.ground_dir, os.listdir(os.path.abspath(args.ground_dir))[0]),

              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    if args.ws_dir:
        params.extend(["--weakly_supervised_path", os.path.join(args.ws_dir, filename.replace("HR", "LR"))])
    return params


if __name__ == '__main__':
    main()
