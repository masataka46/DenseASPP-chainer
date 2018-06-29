import numpy as np
from PIL import Image
# import utility as Utility
import os
import csv
import random
import utility as util
import chainer.cuda as cuda

class Make_datasets_CityScape():
    def __init__(self, base_dir, img_width, img_height, image_dir, seg_dir, img_width_be_crop, img_height_be_crop,
                 crop_flag=False):

        self.base_dir = base_dir
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_be_crop = img_width_be_crop
        self.img_height_be_crop = img_height_be_crop
        self.dir_img = base_dir + image_dir
        self.dir_seg = base_dir + seg_dir
        self.crop_flag = crop_flag
        # self.file_listX = os.listdir(self.dirX)
        # self.file_listY = os.listdir(self.dirY)
        self.file_list = self.get_file_names(self.dir_img)
        self.file_list.sort()
        self.cityScape_color_chan = np.array([
            [0.0, 0.0, 0.0],#0
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [111.0, 74.0, 0.0],  #
            [81.0, 0.0, 81.0],  #
            [128.0, 64.0, 128.0],  #
            [244.0, 35.0, 232.0],  #
            [250.0, 170.0, 160.0],  #
            [230.0, 150.0, 140.0],  #
            [70.0, 70.0, 70.0],  #
            [102.0, 102.0, 156.0],  #
            [190.0, 153.0, 153.0],  #
            [180.0, 165.0, 180.0],  #
            [150.0, 100.0, 100.0],  #
            [150.0, 120.0, 90.0],  #
            [153.0, 153.0, 153.0],  #
            [153.0, 153.0, 153.0],  #
            [250.0, 170.0, 30.0],  #
            [220.0, 220.0, 0.0],  #
            [107.0, 142.0, 35.0],  #
            [152.0, 251.0, 152.0],  #
            [70.0, 130.0, 180.0],  #
            [220.0, 20.0, 60.0],  #
            [255.0, 0.0, 0.0],  #
            [0.0, 0.0, 142.0],  #
            [0.0, 0.0, 70.0],#
            [0.0, 60.0, 100.0],  #
            [0.0, 0.0, 90.0],#
            [0.0, 0.0, 110.0],#
            [0.0, 80.0, 100.0],  #
            [0.0, 0.0, 230.0],#
            [119.0, 11.0, 32.0],#
            [0.0, 0.0, 142.0]  #
            ], dtype=np.float32
            )
        print("self.cityScape_color_chan[3]", self.cityScape_color_chan[3])

        self.image_file_num = len(self.file_list)
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_listX)", len(self.file_list))
        print("self.image_fileX_num", self.image_file_num)


    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files


    def get_only_img(self, list, extension):
        list_mod = []
        for y in list:
            if (y[-9:] == extension): #only .png
                list_mod.append(y)
        return list_mod


    def read_1_data(self, dir, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                    margin_W_batch, crop_flag, seg_flag=False):
        images = []
        for num, filename in enumerate(filename_list):
            if seg_flag:
                # ex) bremen_000000_000019_leftImg8bit.png to bremen_000000_000019_gtFine_color.png
                # print("filename, ", filename)
                _ , dir_name, file_name_only = filename.rsplit("/", 2)

                str_base, _ = file_name_only.rsplit("_",1)
                filename_seg = str_base + "_gtFine_labelIds.png"
                # print("dir + dir_name + '/' + filename_seg", dir + dir_name + '/' + filename_seg)
                # pilIn = Image.open(dir + filename_seg)
                pilIn = Image.open(dir + dir_name + '/' + filename_seg)
            else:
                # pilIn = Image.open(dir + filename)
                pilIn = Image.open(filename)

            if crop_flag:
                pilIn = pilIn.resize((width_be_crop, height_be_crop))
                pilResize = self.crop_img(pilIn, width, height, margin_W_batch[num], margin_H_batch[num])
                # print("pilResize.size", pilResize.size)
            else:
                pilResize = pilIn.resize((width, height))
            # image = np.asarray(pilResize, dtype=np.float32)

            if seg_flag:
                image = np.asarray(pilResize, dtype=np.int32)
                # image = image[:,:,:3]
                # image = self.convert_color_to_indexInt(image)
                image_t = image
            else:
                image = np.asarray(pilResize, dtype=np.float32)
                image_t = np.transpose(image, (2, 0, 1))
            # except:
            #     print("filename =", filename)
            #     image_t = image.reshape(image.shape[0], image.shape[1], 1)
            #     image_t = np.tile(image_t, (1, 1, 3))
            #     image_t = np.transpose(image_t, (2, 0, 1))
            images.append(image_t)

        return np.asarray(images)


    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0
        # data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm


    def crop_img(self, data, output_img_W, output_img_H, margin_W, margin_H):
        cropped_img = data.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
        return cropped_img


    def read_1_data_and_convert_RGB(self, dir, filename_list, extension, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename[0] + extension).convert('RGB')
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize)
            image_t = np.transpose(image, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)


    def write_data_to_img(self, dir, np_arrays, extension):

        for num, np_array in enumerate(np_arrays):
            pil_img = Image.fromarray(np_array)
            pil_img.save(dir + 'debug_' + str(num) + extension)


    def convert_color_to_30chan(self, data): # for cityScape dataset when use Tensorflow
        # print("data.shape", data.shape)
        # print("self.cityScape_color_chan.shape", self.cityScape_color_chan.shape)
        d_mod = np.zeros((data.shape[0], data.shape[1], 30), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    # print("ele.shape", ele.shape)
                    # print("chan.shape", chan.shape)
                    if np.allclose(chan, ele):
                        d_mod[h][w][num] = 1.0
        return d_mod

    def convert_30chan_to_color(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(ele):
                    if chan == 1.0:
                        d_mod[h][w] = self.cityScape_color_chan[num]
        return d_mod

    def convert_color_to_indexInt(self, data): # for cityScape dataset when use Chainer
        d_mod = np.zeros((data.shape[0], data.shape[1]), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    if np.allclose(chan, ele):
                        d_mod[h][w]= num
        return d_mod

    def convert_indexInt_to_color(self, data):
        # print("data.shape", data.shape)
        # print("data[0][0]", data[0][0])
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                # print("ele", ele)
                # print("type(ele), ", type(ele))
                ele_np = cuda.to_cpu(ele)
                # print("type(ele_np)", type(ele_np))
                # print("self.cityScape_color_chan[ele]", self.cityScape_color_chan[ele])
                d_mod[h][w] = self.cityScape_color_chan[ele_np]

                # d_mod[h][w][0] = self.cityScape_color_chan[ele][0]
                # d_mod[h][w][1] = self.cityScape_color_chan[ele][1]
                # d_mod[h][w][2] = self.cityScape_color_chan[ele][2]

        return d_mod


    def convert_to_0_1_class_(self, d):
        d_mod = np.zeros((d.shape[0], d.shape[1], d.shape[2], self.class_num), dtype=np.float32)

        for num, image1 in enumerate(d):
            for h, row in enumerate(image1):
                for w, ele in enumerate(row):
                    if int(ele) == 255:#border
                    # if int(ele) == 255 or int(ele) == 0:#border and backgrounds
                        # d_mod[num][h][w][20] = 1.0
                        continue
                    # d_mod[num][h][w][int(ele) - 1] = 1.0
                    d_mod[num][h][w][int(ele)] = 1.0
        return d_mod


    def make_data_for_1_epoch(self):
        self.image_files_1_epoch = random.sample(self.file_list, self.image_file_num)
        self.margin_H = np.random.randint(0, (self.img_height_be_crop - self.img_height + 1), self.image_file_num)
        self.margin_W = np.random.randint(0, (self.img_width_be_crop - self.img_width + 1), self.image_file_num)
        # print("self.margin_H", self.margin_H)
        # print("self.margin_W", self.margin_W)


    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):

        data_batch = self.image_files_1_epoch[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]

        images = self.read_1_data(self.dir_img, data_batch, self.img_width, self.img_height, self.img_width_be_crop,
                                   self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        # imagesY = self.read_1_data(self.dirY, data_batchY, self.img_width, self.img_height, self.img_width_be_crop,
        #                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        segs = self.read_1_data(self.dir_seg, data_batch, self.img_width, self.img_height,
                self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        # imagesY_seg = self.read_1_data(self.dirY_seg, data_batchY, self.img_width, self.img_height,
        #         self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        images_n = self.normalize_data(images)
        # imagesY_n = self.normalize_data(imagesY)

        # imagesX_n_seg = self.normalize_data(imagesX_seg)
        # imagesY_n_seg = self.normalize_data(imagesY_seg)

        # labels_0_1 = self.convert_to_0_1_class_(labels)
        return images_n, segs


    def make_img_from_label(self, labels, epoch):#labels=(first_number, last_number + 1)
        labels_train = self.train_files_1_epoch[labels[0]:labels[1]]
        labels_val = self.list_val_files[labels[0]:labels[1]]
        labels_train_val = labels_train + labels_val
        labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, labels_train_val, '.png', self.img_width, self.img_height)
        self.write_data_to_img('debug/label_' + str(epoch) + '_',  labels_img_np, '.png')


    def make_img_from_prob(self, probs, epoch):#probs=(data, height, width)..0-20 value
        # print("probs[0]", probs[0])
        print("probs[0].shape", probs[0].shape)
        probs_RGB = util.convert_indexColor_to_RGB(probs)
        # labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, probs_RGB, '.jpg', self.img_width, self.img_height)
        self.write_data_to_img('debug/prob_' + str(epoch), probs_RGB, '.jpg')


    def get_concat_img_h(self, img1, img2):
        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        return dst

    def get_concat_img_w(self, img1, img2):
        dst = Image.new('RGB', (img1.width, img1.height + img2.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (0, img1.height))
        return dst

    def make_img_from_seg_prob(self, img_X, probsX, segsX, out_image_dir, epoch, LOG_FILE_NAME):
        print("in make_img_from_seg_prob()")
        print("img_X.shape, ", img_X.shape)
        # print("img_Y.shape, ", img_Y.shape)
        print("probsX.shape, ", probsX.shape)
        # print("probsY.shape, ", probsY.shape)
        print("segsX.shape, ", segsX.shape)
        # print("segsY.shape, ", segsY.shape)

        # # probs_transX = np.transpose(probsX, (0, 2, 3, 1))
        # probs_argmaxX = np.argmax(probsX, axis=3)
        # print("probs_argmaxX.shape", probs_argmaxX.shape)
        # # probs_transY = np.transpose(probsY, (0, 2, 3, 1))
        # probs_argmaxY = np.argmax(probsY, axis=3)
        # print("probs_argmaxY.shape", probs_argmaxY.shape)
        # segs_transX = np.transpose(segsX, (0, 2, 3, 1))
        # segs_argmaxX = np.argmax(segsX, axis=3)
        # print("segs_argmaxX.shape", segs_argmaxX.shape)
        # segs_argmaxY = np.argmax(segsY, axis=3)
        # print("segs_argmaxY.shape", segs_argmaxY.shape)

        probs_segX = []
        for num, prob in enumerate(probsX):
            probs_segX.append(self.convert_indexInt_to_color(prob))
        probs_segX_np = np.array(probs_segX, dtype=np.float32)

        # probs_segY = []
        # for num, prob in enumerate(probsY):
        #     probs_segY.append(self.convert_indexInt_to_color(prob))
        # probs_segY_np = np.array(probs_segY, dtype=np.float32)

        segX = []
        for num, prob in enumerate(segsX):
            segX.append(self.convert_indexInt_to_color(prob))
        segX_np = np.array(segX, dtype=np.float32)

        # segY = []
        # for num, prob in enumerate(segsY):
        #     segY.append(self.convert_indexInt_to_color(prob))
        # segY_np = np.array(segY, dtype=np.float32)


        #img_X, img_X2Y, img_X2Y2X, img_Y, img_Y2X, img_Y2X2Y, out_image_dir, epoch, log_file_name
        util.make_output_img_seg(img_X, probs_segX_np, segX_np, out_image_dir, epoch, LOG_FILE_NAME)


if __name__ == '__main__':
    #debug
    FILE_NAME = 'bremen_000309_000019_gtFine_color.png'
    base_dir = '/media/webfarmer/HDCZ-UT/dataset/cityScape/'
    image_dirX = 'data/leftImg8bit/train/bremen/'
    image_dirY = 'data/leftImg8bit/train/bremen/'
    image_dirX_seg = 'gtFine/train/bremen/'
    image_dirY_seg = 'gtFine/train/bremen/'

    img_width = 200
    img_height = 200
    img_be_crop_width = 400
    img_be_crop_height = 200
    '''
    make_datasets_CityScape = Make_datasets_CityScape(base_dir, img_width, img_height, image_dirX, image_dirX,
                                image_dirX_seg, image_dirX_seg, img_be_crop_width, img_be_crop_height, crop_flag=True)

    make_datasets_CityScape.make_data_for_1_epoch()
    imagesX, imagesY, imagesX_seg, imagesY_seg = make_datasets_CityScape.get_data_for_1_batch(0, 1, train_FLAG=True)
    print("imagesX.shape", imagesX.shape)
    print("imagesX.dtype", imagesX.dtype)
    print("imagesX[0][2][10][10]", imagesX[0][2][10][10])
    print("imagesY.shape", imagesY.shape)
    print("imagesY.dtype", imagesY.dtype)
    print("imagesY[0][2][10][10]", imagesY[0][2][10][10])
    print("np.max(imagesY)", np.max(imagesY))
    print("np.min(imagesY)", np.min(imagesY))
    print("imagesX_seg.shape", imagesX_seg.shape)
    print("imagesX_seg.dtype", imagesX_seg.dtype)
    print("imagesX_seg[0]", imagesX_seg[0])
    print("imagesY_seg.shape", imagesY_seg.shape)
    print("imagesY_seg.dtype", imagesY_seg.dtype)
    print("self.file_listX[0]", make_datasets_CityScape.file_listX[0])
    print("self.file_listX[10]", make_datasets_CityScape.file_listX[10])

    image_debug_seg1 = (imagesX_seg[0])
    image_debug_seg2 = make_datasets_CityScape.convert_indexInt_to_color(image_debug_seg1)
    image_debug_seg3 = Image.fromarray(image_debug_seg2.astype(np.uint8))
    image_debug_ori = Image.fromarray(((imagesX[0] + 1.0) * 127.5).transpose(1, 2, 0).astype(np.uint8))
    image_concat = make_datasets_CityScape.get_concat_img_h(image_debug_ori, image_debug_seg3)
    image_debug_seg1y = (imagesY_seg[0])
    image_debug_seg2y = make_datasets_CityScape.convert_indexInt_to_color(image_debug_seg1y)
    image_debug_seg3y = Image.fromarray(image_debug_seg2y.astype(np.uint8))
    image_debug_oriy = Image.fromarray(((imagesY[0] + 1.0) * 127.5).transpose(1, 2, 0).astype(np.uint8))
    image_concaty = make_datasets_CityScape.get_concat_img_h(image_debug_oriy, image_debug_seg3y)
    image_big = make_datasets_CityScape.get_concat_img_w(image_concat, image_concaty)

    image_big.show()
    '''