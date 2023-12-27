import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec

# from utils import *
# from network_origin import SwinIR as swinir
from torch.nn.parallel import DataParallel, DistributedDataParallel

from imresize import imresize
from gkernel import generate_kernel
import numpy as np

import time
import imageio
from utils import *

import glob
import scipy.io
from argparse import ArgumentParser
import yaml
from models.elan_network import create_model




# from network_origin import SwinIR as swinir
from models.network_mbconv_elab_swinir import SwinIR as swinir
# from models.network_modified_elab import SwinIR as swinir



class Test(object):
    def __init__(self, model, model_path, save_path, kernel, scale, method_num, num_of_adaptation):
        methods=['direct', 'direct', 'bicubic', 'direct']
        self.save_results=True
        # self.max_iters=num_of_adaptation
        self.max_iters = 10
        self.display_iter = 1

        self.upscale_method= 'cubic'
        self.noise_level = 0.0

        # self.back_projection=False
        self.back_projection=True

        self.back_projection_iters=10

        self.model_path=model_path
        self.save_path=save_path
        self.method_num=method_num

        self.ds_method=methods[self.method_num]

        self.kernel = kernel
        self.scale=scale
        self.scale_factors = [self.scale, self.scale]
        self.device = torch.device('cuda')
        self.learning_rate = 2e-2

        self.net = model
        self.net = self.net.to(self.device)
        # self.net = swinir(upscale=2,
        #            in_chans=3,
        #            img_size=64,
        #            window_size=16,
        #            img_range=1.0,
        #            depths=[6,6,6,6],
        #            embed_dim=60,
        #            num_heads=[6,6,6,6],
        #            mlp_ratio=2,
        #            upsampler="pixelshuffledirect",
        #            resi_connection="1conv")
        # self.net = self.net.to(self.device)

        # self.build_network(conf)

    def initialize(self):
        # self.sess.run(self.init)

        # self.loader.restore(self.sess, self.model_path)
        
        print('load pretrained model: {}!'.format(self.model_path))
        ckpt = torch.load(self.model_path)
        self.net.load_state_dict(ckpt['model_state_dict'])

        # param_key='params'
        # state_dict = torch.load(self.model_path)
        # if param_key in state_dict.keys():
        #     state_dict = state_dict[param_key]
        # self.net.load_state_dict(state_dict, strict=True)

        print('=============== Load Meta-trained Model parameters... ==============')

        self.loss = [None] * self.max_iters
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.psnr=[]
        self.iter = 0

    def __call__(self, img, gt, img_name):
        self.img=img
        self.gt = modcrop(gt, self.scale)

        self.img_name=img_name

        print('** Start Adaptation for X', self.scale, os.path.basename(self.img_name), ' **')
        # Initialize network
        self.initialize()

        self.sf = np.array(self.scale_factors)
        self.output_shape = np.uint(np.ceil(np.array(self.img.shape[0:2]) * self.scale))

        # Train the network

          
        self.quick_test()

        print('[*] Baseline ')
        self.train()

        post_processed_output = self.final_test()
        
        # post_processed_output = self.quick_test()
        
        '''
        if self.save_results:
            if not os.path.exists('%s/%02d' % (self.save_path, self.max_iters)):
                os.makedirs('%s/%02d' % (self.save_path, self.max_iters))
            ####################################################################################    
            # self.here_img_name = os.path.basename(self.img_name)[:-4]
            # judge_path = os.path.join(self.save_path, self.max_iters, self.here_img_name + '.png')
            # while os.path.isfile(judge_path):
            #     self.here_img_name += 'a'
            # imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, self.here_img_name),
            #             post_processed_output)
            ####################################################################################
            
                

            # imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, os.path.basename(self.img_name)[:-4]),
            imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, os.path.basename(self.img_name)[:-4]),
                                  post_processed_output)
        '''  

        print('** Done Adaptation for X', self.scale, os.path.basename(self.img_name),', PSNR: %.4f' % self.psnr[-1], ' **')
        print('')

        return post_processed_output, self.psnr

    def train(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.net.parameters(),lr = self.learning_rate)

        self.hr_father = self.img
        self.lr_son = imresize(self.img, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method)
        self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.)

        ###################################################################################################
        # height, weight, _ = self.img.shape
        # new_h = height//2
        # new_w = weight//2
        # interval = min(new_h, new_w)
        # if interval % 2 != 0:
        #     interval += 1


        t1=time.time()
        for self.iter in range(self.max_iters):
 
            # random1 = torch.randint(0,new_h,(1,1))
            # random2 = torch.randint(0,new_w,(1,1))
            # random1 = torch.randint(0,interval-1,(1,1))
            # random2 = torch.randint(0,interval-1,(1,1))

            # self.patch_input = self.img[random1:random1+new_h, random2:random2+new_w, ...]
            # self.patch_input = self.img[random1:random1+interval, random2:random2+interval, ...]

            # if self.patch_input.shape[1] == 57:
            #     print("stop")


            # self.hr_father = self.patch_input
            # self.lr_son = imresize(self.patch_input, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method)
            # self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.)
            

            if self.method_num == 0:
                '''direct'''
                if self.iter==0:
                    self.learning_rate=2e-2
                elif self.iter < 4:
                    self.learning_rate=1e-2
                else:
                    self.learning_rate=5e-3

            elif self.method_num == 1:
                '''Multi-scale'''
                if self.iter < 3:
                    self.learning_rate=1e-2
                else:
                    self.learning_rate=5e-3

            elif self.method_num == 2:
                '''bicubic'''
                if self.iter == 0:
                    self.learning_rate = 0.01
                elif self.iter < 3:
                    self.learning_rate = 0.01
                else:
                    self.learning_rate = 0.001

            elif self.method_num == 3:
                ''''scale 4'''
                if self.iter ==0:
                    self.learning_rate=1e-2
                elif self.iter < 5:
                    self.learning_rate=5e-3
                else:
                    self.learning_rate=1e-3

            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, criterion, optimizer)

            # Display information
            if self.iter % self.display_iter == 0:
                print('Scale: ', self.scale, ', iteration: ', (self.iter+1), ', loss: ', self.loss[self.iter])

            # Test network during adaptation

            # if self.iter % self.display_iter == 0:
            #     output=self.quick_test()

            # if self.iter==0:
            #     imageio.imsave('%s/%02d/01/%s.png' % (self.save_path, self.method_num, os.path.basename(self.img_name)[:-4]), output)
            # if self.iter==9:
            #     imageio.imsave('%s/%02d/10/%s_%d.png' % (self.save_path, self.method_num, os.path.basename(self.img_name)[:-4], self.iter), output)

        t2 = time.time()
        print('%.2f seconds' % (t2 - t1))

    def forward_pass(self, input, output_shape=None):
        '''
        ILR = imresize(input, self.scale, output_shape, self.upscale_method)
        feed_dict = {self.input : ILR[None,:,:,:]}

        output_=self.sess.run(self.output, feed_dict)
        '''
        input = torch.from_numpy(input.copy()).permute(2,0,1).unsqueeze_(0).type(torch.FloatTensor).to(self.device)
        output_ = self.net(input).permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(np.squeeze(output_), 0., 1.)

    def forward_backward_pass(self, input, hr_father, criterion, optimizer):
        input = torch.from_numpy(input).permute(2,0,1).unsqueeze_(0).type(torch.FloatTensor).to(self.device)
        hr_father = torch.from_numpy(hr_father).permute(2,0,1).unsqueeze_(0).type(torch.FloatTensor).to(self.device)
        train_output = self.net(input)

        # print(train_output.shape)
        # print(hr_father.shape)


        if hr_father.shape == train_output.shape:
            loss = criterion(hr_father, train_output)
        else:
            # loss = criterion(hr_father, train_output[:,:,:hr_father.shape[2],:])
            loss = criterion(hr_father, train_output[:,:,:hr_father.shape[2],:hr_father.shape[3]])

            



        # loss = criterion(hr_father, train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.loss[self.iter] = loss
        output_ = train_output.permute(0,2,3,1).detach().cpu().numpy()

        '''
        ILR = imresize(input, self.scale, hr_father.shape, self.upscale_method)

        HR = hr_father[None, :, :, :]

        # Create feed dict
        feed_dict = {self.input: ILR[None,:,:,:], self.label: HR, self.lr_decay: self.learning_rate}
        
        # Run network
        _, self.loss[self.iter], train_output = self.sess.run([self.opt, self.loss_t, self.output], feed_dict=feed_dict)
        '''
        
        # return np.clip(np.squeeze(train_output), 0., 1.)
        return np.clip(np.squeeze(output_), 0., 1.)


    def hr2lr(self, hr):
        lr = imresize(hr, 1.0 / self.scale, kernel=self.kernel, ds_method=self.ds_method)
        return np.clip(lr + np.random.randn(*lr.shape) * self.noise_level, 0., 1.)

    def quick_test(self):
        # 1. True MSE
        self.sr = self.forward_pass(self.img, self.gt.shape)

        if not self.sr.shape == self.gt.shape:
            self.sr = self.sr[:self.gt.shape[0], :self.gt.shape[1],:]
        self.mse = self.mse + [np.mean((self.gt - self.sr)**2)]

        '''Shave'''
        scale=int(self.scale)
        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale])
        '''
        if self.save_results:
            if not os.path.exists('%s/%02d' % (self.save_path, self.max_iters)):
                os.makedirs('%s/%02d' % (self.save_path, self.max_iters))

            # self.here_img_name = os.path.basename(self.img_name)[:-4]
            # judge_path = '%s/%02d/%s.png' % (self.save_path, self.max_iters, self.here_img_name)
            # while os.path.isfile(judge_path):
            #     self.here_img_name += 'a'

            imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, os.path.basename(self.img_name)[:-4] + '_lwsr'),
            # imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, self.here_img_name + '_lwsr'),
        
        # imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, os.path.basename(self.img_name)[:-4] + '_lwsr'),

                                self.sr)
        '''
        
        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)), rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8)))
        self.psnr.append(PSNR)

        # 2. Reconstruction MSE
        self.reconstruct_output = self.forward_pass(self.hr2lr(self.img), self.img.shape)

        if self.img.shape == self.reconstruct_output.shape:
            self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))
        else:
            self.mse_rec.append(np.mean((self.img - self.reconstruct_output[:self.img.shape[0], :self.img.shape[1], :self.img.shape[2]])**2))
            


        '''这里加入了img和reconstruct_output的shape的比较'''

        
        # self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))

        processed_output=np.round(np.clip(self.sr*255, 0., 255.)).astype(np.uint8)

        print('iteration: ', self.iter, 'recon mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)

        return processed_output
    
    def my_back_projection(self, y_sr, y_lr, down_kernel, up_kernel, sf=None, ds_method='direct'):
    # print(y_sr.shape)
    # print(y_lr.shape)
        tmp = y_lr - imresize(y_sr, scale=1.0/sf, output_shape=y_lr.shape, kernel=down_kernel, ds_method=ds_method)
        tmp = torch.from_numpy(tmp).permute(2,0,1).unsqueeze_(0).type(torch.FloatTensor).to(self.device)
        result_tmp = self.net(tmp).permute(0,2,3,1).squeeze()
        y_sr += result_tmp.detach().cpu().numpy()



    def final_test(self):
        outputs = []
        for k in range(0, 8, 1):
            test_input = np.rot90(self.img, k) if k < 4 else np.fliplr(np.rot90(self.img, k))
            tmp_output = self.forward_pass(test_input, self.gt.shape)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)

            if self.back_projection == True:
                for bp_iter in range(self.back_projection_iters):
                    # output = back_projection(output, self.img, down_kernel=self.kernel,
                    tmp_output = back_projection(tmp_output, self.img, down_kernel=self.kernel,
                                                    up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)
            outputs.append(tmp_output)
        almost_final_sr = np.median(outputs, 0)
        for bp_iter in range(self.back_projection_iters):
                    # output = back_projection(output, self.img, down_kernel=self.kernel,
                    almost_final_sr = back_projection(almost_final_sr, self.img, down_kernel=self.kernel,
                                                    up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)
        processed_output=np.round(np.clip(almost_final_sr*255, 0., 255.)).astype(np.uint8)
        
        
            


        










        # output = self.forward_pass(self.img, self.gt.shape)
        # if self.back_projection == True:
        #     for bp_iter in range(self.back_projection_iters):
        #         # output = back_projection(output, self.img, down_kernel=self.kernel,
        #         output = back_projection(output, self.img, down_kernel=self.kernel,
        #                                           up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)

        # processed_output=np.round(np.clip(output*255, 0., 255.)).astype(np.uint8)

        '''Shave'''
        scale=int(self.scale)
        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(processed_output)[scale:-scale, scale:-scale])

        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)),
        #           rgb2y(processed_output))

        self.psnr.append(PSNR)

        return processed_output




def main():
    parser=ArgumentParser()

    # Global
    parser.add_argument('--gpu', type=str, dest='gpu', default='0')

    # For Meta-test
    parser.add_argument('--inputpath', type=str, dest='inputpath', default='TestSet/Set5/g13/LR/')
    parser.add_argument('--gtpath', type=str, dest='gtpath', default='TestSet/Set5/GT_crop/')
    parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='TestSet/Set5/g13/kernel.mat')
    parser.add_argument('--savepath', type=str, dest='savepath', default='results/Set5')
    parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=0)
    parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)

    # For Meta-Training
    parser.add_argument('--trial', type=int, dest='trial', default=0)
    parser.add_argument('--step', type=int, dest='step', default=0)
    parser.add_argument('--train', dest='is_train', default=False, action='store_true')
    parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
    
    args= parser.parse_args()

    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

    if args.model==0:
        print('Direct Downscaling, Scaling factor x2 Model')
        # model_path = 'Model/Directx2'
        # model_path = '/home/shiminghao/smh/trytry/test_results/checkpoint_9500_9.pth'
        model_path = args.pretrain
        
        # model_path = '/home/shiminghao/smh/trytry/435000_G.pth'
        # model_path = '/home/shiminghao/smh/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
        
   
        # model_path = '/home/shiminghao/smh/trytry/results_test/checkpoint_7000_0.pth'
        # model_path = '/home/sheminghao/smh/trytry/results/checkpoint_200_9.pth'
    elif args.model ==1:
        print('Direct Downscaling, Multi-scale Model')
        model_path = 'Model/Multi-scale'
    elif args.model ==2:
        print('Bicubic Downscaling, Scaling factor x2 Model')
        model_path = 'Model/Bicubicx2'
    elif args.model ==3:
        print('Direct Downscaling, Scaling factor x4 Model')
        model_path = 'Model/Directx4'
    
    model = create_model(args)
    

    img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
    gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))
    # img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.JPEG')))
    # gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.JPEG')))

    scale=2.0

    try:
        # kernel=scipy.io.loadmat(args.kernelpath)['kernel']
        kernel=scipy.io.loadmat(args.kernelpath)['Kernel']
    except:
        kernel='cubic'

    Tester=Test(model, model_path, args.savepath, kernel, scale, args.model, args.num_of_adaptation)
    P=[]
    for i in range(len(img_path)):
        img=imread(img_path[i])
        gt=imread(gt_path[i])

        _, pp =Tester(img, gt, img_path[i])

        P.append(pp)
    
    
    avg_PSNR=np.mean(P, 0)
    print(avg_PSNR)


    print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))
    # print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % (avg_PSNR, avg_PSNR))







if __name__ == '__main__':
    main()