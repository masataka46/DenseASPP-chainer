import numpy as np
import os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
# from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
# from chainer.training import extensions
# from PIL import Image
import utility as Utility
from make_datasets import Make_datasets_CityScape
import argparse
import chainer.cuda as cuda


def parser():
    parser = argparse.ArgumentParser(description='analyse oyster images')
    parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log01', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='epoch')
    parser.add_argument('--base_dir', '-bd', type=str, default='/media/webfarmer/HDCZ-UT/dataset/cityScape/',
                        help='base directory name of data-sets')
    parser.add_argument('--img_dir', '-id', type=str, default='data/leftImg8bit/train/', help='directory name of image data')
    parser.add_argument('--seg_dir', '-sd', type=str, default='gtFine/train/', help='directory name of data X')
    parser.add_argument('--input_image_size', '-iim', type=int, default=256, help='input image size, only 256 or 128')

    return parser.parse_args()
args = parser()


#global variants
BATCH_SIZE = args.batchsize
N_EPOCH = args.epoch
WEIGHT_DECAY = 0.00001
BASE_CHANNEL = 512
IMG_SIZE = args.input_image_size
IMG_SIZE_BE_CROP_W = 512
IMG_SIZE_BE_CROP_H = 256
BASE_DIR = args.base_dir
LOG_FILE_NAME = args.log_file_name
CLASS_NUM = 35 #cityScape dataset
keep_prob_rate = 0.5
OUT_PUT_IMG_NUM = 6
seed = 1234
np.random.seed(seed=seed)

out_image_dir = './out_images_DenseASPP' #output image file
out_model_dir = './out_models_DenseASPP' #output model file

try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    print("mkdir error")
    pass
#base_dir, img_width, img_height, image_dir, seg_dir, img_width_be_crop, img_height_be_crop, crop_flag=False
make_data = Make_datasets_CityScape(BASE_DIR, IMG_SIZE, IMG_SIZE, args.img_dir, args.seg_dir, IMG_SIZE_BE_CROP_W,
                                    IMG_SIZE_BE_CROP_H, crop_flag=True)
iniW = chainer.initializers.Normal(scale=0.02)


#Segmentor Y-----------------------------------------------------------------
class DenseASPP(chainer.Chain):
    def __init__(self):
        super(DenseASPP, self).__init__(

            # First Convolution
            convF1=L.Convolution2D(3, BASE_CHANNEL, ksize=3, stride=1, pad=1, initialW=iniW),  # 128x128 to 128x128
            convF2=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL, ksize=3, stride=1, pad=1, initialW=iniW),  # 128x128 to 128x128
            convF3=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL, ksize=3, stride=1, pad=1, initialW=iniW),

            conv1x1_D3=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL // 2, ksize=1, stride=1, pad=0, initialW=iniW),  # 128x128 to 128x128

            dilate_conv3=L.DilatedConvolution2D(BASE_CHANNEL // 2, BASE_CHANNEL // 8,
                                                ksize=3, stride=1, pad=2, dilate=2, nobias=False, initialW=None,
                                                initial_bias=None),

            conv1x1_D6=L.Convolution2D(BASE_CHANNEL * 9 // 8, BASE_CHANNEL // 2, ksize=1, stride=1, pad=0, initialW=iniW),

            dilate_conv6 = L.DilatedConvolution2D(BASE_CHANNEL // 2, BASE_CHANNEL // 8,
                                              ksize=3, stride=1, pad=5, dilate=5, nobias=False, initialW=None,
                                              initial_bias=None),

            conv1x1_D12=L.Convolution2D(BASE_CHANNEL * 10 // 8, BASE_CHANNEL // 2, ksize=1, stride=1, pad=0, initialW=iniW),

            dilate_conv12 = L.DilatedConvolution2D(BASE_CHANNEL // 2, BASE_CHANNEL // 8,
                                               ksize=3, stride=1, pad=11, dilate=11, nobias=False, initialW=None,
                                               initial_bias=None),

            conv1x1_D18=L.Convolution2D(BASE_CHANNEL * 11 // 8, BASE_CHANNEL // 2,ksize=1, stride=1, pad=0,
                                        initialW=iniW),

            dilate_conv18 = L.DilatedConvolution2D(BASE_CHANNEL // 2, BASE_CHANNEL // 8,
                                               ksize=3, stride=1, pad=17, dilate=17, nobias=False, initialW=None,
                                               initial_bias=None),

            conv1x1_D24=L.Convolution2D(BASE_CHANNEL * 12 // 8, BASE_CHANNEL // 2, ksize=1, stride=1, pad=0,
                                        initialW=iniW),

            dilate_conv24 = L.DilatedConvolution2D(BASE_CHANNEL // 2, BASE_CHANNEL // 8,
                                               ksize=3, stride=1, pad=23, dilate=23, nobias=False, initialW=None,
                                               initial_bias=None),

            convL=L.Convolution2D(BASE_CHANNEL * 13 // 8, CLASS_NUM, ksize=3, stride=1, pad=1, initialW=iniW),  # 128x128 to 128x128

            # batch normalization
            bnF1=L.BatchNormalization(3),
            bnF2=L.BatchNormalization(BASE_CHANNEL),
            bnF3=L.BatchNormalization(BASE_CHANNEL),

            bn1x1_D3=L.BatchNormalization(BASE_CHANNEL),
            bnD3=L.BatchNormalization(BASE_CHANNEL // 2),
            bn1x1_D6=L.BatchNormalization(BASE_CHANNEL * 9 // 8),
            bnD6=L.BatchNormalization(BASE_CHANNEL // 2),
            bn1x1_D12=L.BatchNormalization(BASE_CHANNEL * 10 // 8),
            bnD12=L.BatchNormalization(BASE_CHANNEL // 2),
            bn1x1_D18=L.BatchNormalization(BASE_CHANNEL * 11 // 8),
            bnD18=L.BatchNormalization(BASE_CHANNEL // 2),
            bn1x1_D24=L.BatchNormalization(BASE_CHANNEL * 12 // 8),
            bnD24=L.BatchNormalization(BASE_CHANNEL // 2),

            bnL=L.BatchNormalization(BASE_CHANNEL * 13 // 8),

        )

    def __call__(self, x, train=True):
        # First Convolution
        c0 = self.bnF1(x)
        c0 = self.convF1(c0)
        c0 = F.relu(c0)

        c0 = self.bnF2(c0)
        c0 = self.convF2(c0)
        c0 = F.relu(c0)

        c0 = self.bnF3(c0)
        c0 = self.convF3(c0)
        c0 = F.relu(c0)

        # Atrous Convolution size 3
        cn3 = self.bn1x1_D3(c0)
        cn3 = self.conv1x1_D3(cn3)
        cn3 = self.bnD3(cn3)
        cn3 = self.dilate_conv6(cn3)
        cn3 = F.relu(cn3)

        # Atrous Convolution size 6
        cn3_con = F.concat((c0, cn3), axis=1)
        cn6 = self.bn1x1_D6(cn3_con)
        cn6 = self.conv1x1_D6(cn6)
        cn6 = self.bnD6(cn6)
        cn6 = self.dilate_conv6(cn6)
        cn6 = F.relu(cn6)

        # Atrous Convolution size 12
        cn6_con = F.concat((cn3_con, cn6), axis=1)
        cn12 = self.bn1x1_D12(cn6_con)
        cn12 = self.conv1x1_D12(cn12)
        cn12 = self.bnD12(cn12)
        cn12 = self.dilate_conv12(cn12)
        cn12 = F.relu(cn12)

        # Atrous Convolution size 18
        cn12_con = F.concat((cn6_con, cn12), axis=1)
        cn18 = self.bn1x1_D18(cn12_con)
        cn18 = self.conv1x1_D18(cn18)
        cn18 = self.bnD18(cn18)
        cn18 = self.dilate_conv18(cn18)
        cn18 = F.relu(cn18)

        # Atrous Convolution size 24
        cn18_con = F.concat((cn12_con, cn18), axis=1)
        cn24 = self.bn1x1_D24(cn18_con)
        cn24 = self.conv1x1_D24(cn24)
        cn24 = self.bnD24(cn24)
        cn24 = self.dilate_conv24(cn24)
        cn24 = F.relu(cn24)

        # Last convolution
        cn24_con = F.concat((cn18_con, cn24), axis=1)
        cL = self.bnL(cn24_con)
        cL = self.convL(cL)
        out = F.softmax(cL, axis=1)

        return out


model = DenseASPP()
model.to_gpu()
optimizer = optimizers.Adam(alpha=0.0003, beta1=0.5)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))


#training loop
for epoch in range(0, N_EPOCH):
    sum_accuracy = np.float32(0)
    sum_loss = np.float32(0)

    make_data.make_data_for_1_epoch() #shuffle training data
    len_data = make_data.image_file_num

    for i in range(0, len_data, BATCH_SIZE):
        if i % 100 == 0:
            print("i = ", i)
        images_np, segs_np = make_data.get_data_for_1_batch(i, BATCH_SIZE)
        # print("imagesX_np.shape", imagesX_np.shape)
        images_ = Variable(cuda.to_gpu(images_np))
        # images_Y = Variable(cuda.to_gpu(imagesY_np))
        segs_ = Variable(cuda.to_gpu(segs_np))
        # seg_Y = Variable(cuda.to_gpu(imagesY_seg_np))

        prob = model(images_)

        # Loss
        loss = F.softmax_cross_entropy(prob, segs_)

        # Accuracy
        accuracy = F.accuracy(prob, segs_)

        # for print
        sum_loss += loss.data * len(images_np)
        sum_accuracy += accuracy.data * len(images_np)

        # back prop
        model.cleargrads()
        loss.backward()
        optimizer.update()

    print("----------------------------------------------------------------------")
    print("epoch =", epoch , ", Loss =", sum_loss / len_data, ", Accuracy =", sum_accuracy / len_data)

    if epoch % 5 == 0:
        # outupt generated images
            img_input = []
            img_output = []
            seg_t = []
            for i in range(OUT_PUT_IMG_NUM):
                images_np, segs_np = make_data.get_data_for_1_batch(i, 1)

                img_input.append(images_np[0])

                images_ = Variable(cuda.to_gpu(images_np))
                prob = model(images_)
                out_seg_argmax = F.argmax(prob, axis=1)

                img_output.append(out_seg_argmax.data[0])
                seg_t.append(segs_np)

            img_in_np = np.asarray(img_input).transpose((0, 2, 3, 1))
            img_out_cp = np.asarray(img_output)
            seg_t_np = np.asarray(seg_t).transpose((0, 2, 3, 1))

            print('type(img_in_np)', type(img_in_np))
            print('type(img_out_np)', type(img_out_cp))

            seg_t_np_re = seg_t_np.reshape((seg_t_np.shape[0], seg_t_np.shape[1], seg_t_np.shape[2]))

            print('type(img_X2seg_np)', type(img_out_cp))
            img_out_np = cuda.to_cpu(img_out_cp)

            print('type(img_X2seg_np)', type(img_out_np))

            print('type(seg_t_X_np_re)', type(seg_t_np_re))

            make_data.make_img_from_seg_prob(img_in_np, img_out_np, seg_t_np_re, out_image_dir, epoch, LOG_FILE_NAME)
            print("now i =", i)

