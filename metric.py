import mxnet as mx
import numpy as np
import util
import matplotlib.pyplot as plt
import time

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance:    sqrt((u_pred-u_label)^2 + (v_pred-v_label)^2)
    """
    def __init__(self):
        super(EndPointErr, self).__init__('End Point Error')

    def update(self, gt, pred):
        #
        # pred = pred[self.which].asnumpy()
        # gt = gt[self.which].asnumpy()
        #
        # r = pred - gt
        # r = np.power(r, 2)
        # r = np.sqrt(r.sum(axis=1))

        # plt.figure()
        # plt.imshow(pred[-5].asnumpy()[0].transpose(1,2,0))
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(pred[-4].asnumpy()[0].transpose(1,2,0))
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(pred[-3].asnumpy()[0].transpose(1,2,0))
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(pred[-2].asnumpy()[0].transpose(1,2,0))
        # plt.colorbar()
        # plt.waitforbuttonpress()
        # #
        # plt.figure()
        # plt.imshow(pred[-1].asnumpy()[0, 0])
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(pred[-1].asnumpy()[0, 1])
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(pred[-1].asnumpy()[0, 2])
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # util.plot_velocity_vector(pred[-1].asnumpy()[0, 3:].transpose(1,2,0))
        # plt.colorbar()
        # plt.waitforbuttonpress()

        self.sum_metric += pred[0].asnumpy()
        self.num_inst += 1.0

class D1all(mx.metric.EvalMetric):

    """
       residual > 3  and   residual / gt > 0.05   (defined by kitti)
    """

    def __init__(self):
        super(D1all, self).__init__('D1all')

    def update(self, pred, gt,tau = 3):

        pred = pred[0]
        gt = gt[0]

        pred_all = pred.asnumpy()
        gt_all = gt.asnumpy()

        for i in xrange(gt_all.shape[0]):
            self.sum_metric += util.outlier_sum(pred_all[i][0], gt_all[i][0],tau)
        self.num_inst += gt_all.shape[0]
