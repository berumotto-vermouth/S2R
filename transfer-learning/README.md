# MZSR

#
## Brief explanation of contents

```
├── GT: Ground-truth images
├── Input: Input LR images


Rest codes are for the training and test of MZSR.
```

## Guidelines for Codes

**Requisites should be installed beforehand.**

Clone this repo.
```
cd transfer-learning/
```


### Test

Ready for the input data (low-resolution) and corresponding kernel (kernel.mat file.)

[Options]
```
python my_test_kernel.py --gpu [GPU_number] --inputpath [LR path] --gtpath [HR path] --savepath [SR path]  --kernelpath [kernel.mat path] --model [0/1/2/3] --num [1/10]

--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--inputpath: Path of input images [Default: Input/g20/Set5/]
--gtpath: Path of reference images. [Default: GT/Set5/]
--savepath: Path for the output images. [Default: results/Set5]
--kernelpath: Path of the kernel.mat file. [Default: Input/g20/kernel.mat]
--model: [0/1/2/3]
    -> 0: Direct x2
    -> 1: Multi-scale
    -> 2: Bicubic x2
    -> 3: Direct x4
--num: [1/10] The number of adaptation (gradient updates). [Default 1]

```

You may change other minor options in "test.py."
Line 9 to line 17.

The minor options are shown below.
```
self.save_results=True		-> Whether to save results or not.
self.display_iter = 1		-> The interval of information display.
self.noise_level = 0.0		-> You may sometimes add small noise for real-world images.
self.back_projection=False	-> You may also apply back projection algorithm for better results.
self.back_projection_iters=4	-> The number of iteration of back projection.
```

### An example of test codes

```
python main.py --gpu 0 --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath results/Set5 --kernelpath Input/g20/kernel.mat --model 0 --num 1
```


