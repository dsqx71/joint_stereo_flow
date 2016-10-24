from collections import namedtuple
import cv2
import mxnet as mx
import numpy as np
from sklearn import utils
from config import cfg
from random import randint, uniform
import Queue
import multiprocessing as mp
import atexit
import logging
from config import batchsize

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])


class Dataiter(mx.io.DataIter):

    def __init__(self, dataset, batch_shape, label_shapes, augment_ratio, n_thread=40, be_shuffle=True, ctx=ctx[0]):

        super(Dataiter, self).__init__()

        self.batch_size = batch_shape[0]
        self.shapes = batch_shape[1:]
        self.batch_shape = batch_shape
        self.be_shuffle = be_shuffle
        self.current = 0
        self.ctx = ctx

        if self.be_shuffle:
            self.data_dirs = utils.shuffle(dataset.dirs)
        else:
            self.data_dirs = dataset.dirs

        self.num_imgs = len(self.data_dirs)
        self.label_shapes = label_shapes
        self.get_data_function = dataset.get_data
        self.augment_ratio = augment_ratio

        # setting of multi-process
        self.stop_word = '==STOP--'
        self.n_thread = n_thread
        self.worker_proc = None
        self.stop_flag = mp.Value('b', False)
        self.result_queue = mp.Queue(maxsize=self.batch_size*20)
        self.data_queue = mp.Queue()


    @property
    def provide_data(self):
        return [('img1left_data', self.batch_shape), ('img1right_data', self.batch_shape),
                ('img2left_data', self.batch_shape), ('img2right_data', self.batch_shape)] + \
               [ ('label{}'.format(i+1), (self.batch_size, 5) + self.label_shapes[i]) for i in range(len(self.label_shapes))]

    @property
    def provide_label(self):
        return []

    def _thread_start(self):
        # init workers
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=Dataiter._worker,
                                       args=[pid,
                                             self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.get_data_function,
                                             get_augment,
                                             self.augment_ratio,
                                             crop_or_pad,
                                             self.shapes,
                                             self.label_shapes])
                            for pid in xrange(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()
        atexit.register(cleanup)

    def _insert_queue(self):

        for item in self.data_dirs:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in xrange(self.n_thread)]

    def iter_next(self):

        if self.current + self.batch_size > self.num_imgs:
            return False

        self.img1left = []
        self.img1right = []
        self.img2left = []
        self.img2right = []

        self.auxs = []
        self.labels = [[] for i in range(len(self.label_shapes))]

        for i in xrange(self.current, self.current+self.batch_size):

            img1, img2, img3, img4, label, aux = self.result_queue.get()

            self.img1left.append(img1)
            self.img1right.append(img2)
            self.img2left.append(img3)
            self.img2right.append(img4)
            self.auxs.append(aux)
            for j in range(len(self.labels)):
                self.labels[j].append(label[j])
        self.current += self.batch_size
        return True

    def getdata(self):
        return [mx.nd.array(np.asarray(self.img1left).transpose(0, 3, 1, 2), self.ctx),
                mx.nd.array(np.asarray(self.img1right).transpose(0, 3, 1, 2), self.ctx),
                mx.nd.array(np.asarray(self.img2left).transpose(0, 3, 1, 2), self.ctx),
                mx.nd.array(np.asarray(self.img2right).transpose(0, 3, 1, 2), self.ctx)] + \
               [ mx.nd.array(np.asarray(self.labels[i]), self.ctx) for i in range(len(self.label_shapes))]
    @property
    def getaux(self):
        return np.asarray(self.auxs)

    def getlabel(self):

        return []

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, stop_word, stop_flag,get_data_function,get_augment,augment_ratio,
                crop_or_pad,shapes,label_shapes):

        for item in iter(data_queue.get, stop_word):

            if stop_flag == 1:
                break

            img1, img2, img3, img4, dis1, dis2, change, flow, index = get_data_function(item)

            img1 = img1 * 0.00392156862745098
            img2 = img2 * 0.00392156862745098
            img3 = img3 * 0.00392156862745098
            img4 = img4 * 0.00392156862745098

            img1 = (img1 - img1.reshape(-1, 3).mean(axis=0))
            img2 = (img2 - img2.reshape(-1, 3).mean(axis=0))

            img3 = (img3 - img3.reshape(-1, 3).mean(axis=0))
            img4 = (img4 - img4.reshape(-1, 3).mean(axis=0))

            img1, img2, img3, img4, dis1, dis2, change, flow = crop_or_pad(img1, img2, img3, img4, dis1, dis2, change, flow, shapes[1:])

            if uniform(0,1) < augment_ratio:
                img1, img2, img3, img4, dis1, dis2, change, flow = get_augment(img1, img2, img3, img4, dis1, dis2, change,flow)

            label_list = []
            for i in range(len(label_shapes)):

                dis1_temp = np.expand_dims(cv2.resize(dis1, (label_shapes[i][1], label_shapes[i][0])), 2)
                dis2_temp = np.expand_dims(cv2.resize(dis2, (label_shapes[i][1], label_shapes[i][0])), 2)
                change_temp = np.expand_dims(cv2.resize(change, (label_shapes[i][1], label_shapes[i][0])), 2)
                flow_temp = cv2.resize(flow, (label_shapes[i][1], label_shapes[i][0]))
                label = np.concatenate((dis1_temp, dis2_temp, change_temp, flow_temp), 2).transpose(2, 0, 1)
                label_list.append(label)

            result_queue.put((img1, img2, img3, img4, label_list, index))

    def getindex(self):

        return self.current

    def reset(self):
        self.current = 0
        if self.be_shuffle:
            self.data_dirs = utils.shuffle(self.data_dirs)

        self.shutdown()
        self._insert_queue()
        self._thread_start()

    def shutdown(self):

        # shutdown multi-process
        while True:
            try:
                self.result_queue.get(timeout=1)
            except Queue.Empty:
                break
        while True:
            try:
                self.data_queue.get(timeout=1)
            except Queue.Empty:
                break
        self.stop_flag = True
        if self.worker_proc:
            for i, worker in enumerate(self.worker_proc):
                worker.join(timeout=1)
                if worker.is_alive():
                    logging.error('worker {} is join fail'.format(i))
                    worker.terminate()

def crop_or_pad(img1, img2, img3, img4, dis1, dis2, change, flow, shape):

    y_ori, x_ori = img1.shape[:2]
    y, x = shape
    if x == x_ori and y == y_ori:
        return img1, img2, img3, img4, dis1, dis2, flow
    elif y >= y_ori and x >= x_ori:
        # padding
        tmp1 = np.zeros((y, x, 3))
        tmp2 = np.zeros((y, x, 3))
        tmp3 = np.zeros((y, x, 3))
        tmp4 = np.zeros((y, x, 3))
        x_begin = randint(0, x-x_ori)
        y_begin = randint(0, y-y_ori)

        tmp1[y_begin:y_begin+y_ori, x_begin:x_begin + x_ori, :] = img1[:]
        tmp2[y_begin:y_begin+y_ori, x_begin:x_begin + x_ori, :] = img2[:]
        tmp3[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = img3[:]
        tmp4[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = img4[:]

        tmp5 = np.ones((y, x))
        tmp5[:] = np.nan
        tmp5[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = dis1[:]

        tmp6 = np.ones((y, x))
        tmp6[:] = np.nan
        tmp6[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = dis2[:]

        tmp7 = np.ones((y, x))
        tmp7[:] = np.nan
        tmp7[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = change[:]

        tmp8 = np.ones((y, x, 2))
        tmp8[:] = np.nan
        tmp8[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori,:] = flow[:]

        return tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8

    elif y<=y_ori and x <= x_ori:

        # cropping
        x_begin = randint(0, x_ori - x )
        y_begin = randint(0, y_ori - y )
        return img1[y_begin : y_begin + y, x_begin : x_begin + x, :], img2[y_begin : y_begin + y, x_begin:x_begin + x, :], \
               img3[y_begin : y_begin + y, x_begin : x_begin + x, :], img4[y_begin : y_begin + y, x_begin:x_begin + x, :],\
               dis1[y_begin:y_begin+y, x_begin:x_begin+x], dis2[y_begin:y_begin+y, x_begin:x_begin+x],\
               change[y_begin:y_begin+y, x_begin:x_begin+x], flow[y_begin:y_begin+y, x_begin:x_begin+x,:]
    else:
        y_max = max(y,y_ori)
        x_max = max(x,x_ori)
        tmp1 = np.zeros((y_max, x_max, 3))
        tmp2 = np.zeros((y_max, x_max, 3))
        tmp3 = np.zeros((y_max, x_max, 3))
        tmp4 = np.zeros((y_max, x_max, 3))
        x_begin = randint(0, x_max - x_ori)
        y_begin = randint(0, y_max - y_ori)

        tmp1[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = img1[:]
        tmp2[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = img2[:]
        tmp3[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = img3[:]
        tmp4[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = img4[:]

        tmp5 = np.ones((y_max, x_max))
        tmp5[:] = np.nan
        tmp5[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = dis1[:]

        tmp6 = np.ones((y_max, x_max))
        tmp6[:] = np.nan
        tmp6[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = dis2[:]

        tmp7 = np.ones((y_max, x_max, 2))
        tmp7[:] = np.nan
        tmp7[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = change[:]

        tmp8 = np.ones((y_max, x_max, 2))
        tmp8[:] = np.nan
        tmp8[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori, :] = flow[:]

        return tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8


def get_augment(img1, img2, img3, img4, dis1, dis2, change, flow):

    rows, cols, _ = img1.shape
    translation_range = cfg.dataset.translation_range
    gaussian_noise = cfg.dataset.gaussian_noise

    rgb_cof = np.random.uniform(low=cfg.dataset.rgbmul[0], high=cfg.dataset.rgbmul[1], size=3)
    gaussian_noise_scale = uniform(0.0, gaussian_noise)

    tx = randint(int(-img1.shape[1] * translation_range), int(img1.shape[1] * translation_range))
    ty = randint(int(-img1.shape[0] * translation_range), int(img1.shape[0] * translation_range))
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    beta  = uniform(cfg.dataset.beta[0], cfg.dataset.beta[1])
    alpha = uniform(cfg.dataset.alpha[0], cfg.dataset.alpha[1])

    # multiply rgb factor plus guassian noise
    img1 = img1 * rgb_cof + np.random.normal(loc=0.0, scale=gaussian_noise_scale,size=img1.shape)
    img2 = img2 * rgb_cof + np.random.normal(loc=0.0, scale=gaussian_noise_scale,size=img1.shape)
    img3 = img3 * rgb_cof + np.random.normal(loc=0.0, scale=gaussian_noise_scale, size=img1.shape)
    img4 = img4 * rgb_cof + np.random.normal(loc=0.0, scale=gaussian_noise_scale, size=img1.shape)

    # translation
    img1 = cv2.warpAffine(img1, M, (cols, rows))
    img2 = cv2.warpAffine(img2, M, (cols, rows))
    img3 = cv2.warpAffine(img3, M, (cols, rows))
    img4 = cv2.warpAffine(img4, M, (cols, rows))
    dis1 = cv2.warpAffine(dis1, M, (cols, rows))
    dis2 = cv2.warpAffine(dis2, M, (cols, rows))
    change = cv2.warpAffine(change, M, (cols, rows))
    flow = cv2.warpAffine(flow, M, (cols, rows))

    # brightness and contrastness
    img1 = cv2.multiply(img1, np.array([alpha]))
    img1 = cv2.add(img1, np.array([beta]))
    img2 = cv2.multiply(img2, np.array([alpha]))
    img2 = cv2.add(img2, np.array([beta]))
    img3 = cv2.multiply(img3, np.array([alpha]))
    img3 = cv2.add(img3, np.array([beta]))
    img4 = cv2.multiply(img4, np.array([alpha]))
    img4 = cv2.add(img4, np.array([beta]))

    # scaling
    origin_shape = img1.shape[:2]
    if uniform(0, 1) < 0.5:
        y_shape = randint(1.0*origin_shape[0],int(cfg.dataset.scale[1]*origin_shape[0]))
        x_shape = randint(1.0*origin_shape[1],int(cfg.dataset.scale[1]*origin_shape[1]))
    else:
        y_shape = randint(int(cfg.dataset.scale[0] * origin_shape[0]), 1.0 * origin_shape[0])
        x_shape = randint(int(cfg.dataset.scale[0] * origin_shape[1]), 1.0 * origin_shape[1])

    img1 = cv2.resize(img1, (x_shape, y_shape))
    img2 = cv2.resize(img2, (x_shape, y_shape))
    img3 = cv2.resize(img3, (x_shape, y_shape))
    img4 = cv2.resize(img4, (x_shape, y_shape))

    dis1 = dis1 * (x_shape / float(origin_shape[1]))
    dis2 = dis2 * (x_shape / float(origin_shape[1]))
    flow[:, :, 0] = flow[:, :, 0] * (x_shape/float(origin_shape[1]))
    flow[:, :, 1] = flow[:, :, 1] * (y_shape/float(origin_shape[0]))

    dis1 = cv2.resize(dis1, (x_shape, y_shape))
    dis2 = cv2.resize(dis2, (x_shape, y_shape))
    flow = cv2.resize(flow, (x_shape, y_shape))

    img1, img2, img3, img4, dis1, dis2, flow = \
        crop_or_pad(img1, img2, img3, img4, dis1, dis2, flow, batchsize[2:])

    return img1, img2, img3 ,img4 ,dis1 ,dis2, flow


