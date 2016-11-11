import cv2
import numpy as np
from config import cfg
import util
import glob
from skimage import io

class DataSet:

    def __init__(self):
        pass

    @staticmethod
    def shapes():
        """
            original shape
        """
        pass

    @staticmethod
    def get_data(img_dir):
        """
        Parameters
        ----------
        img_dir : a tuple consists of dirs  :
                (img1_left, img1_right, img2_left, img2_right, disparity1, disparity2, disparity_change, optical flow)
        return
        -------
        img1_left , img1_right , img2_left , img2_rigtht, dis1, dis2, flow, valid
        """
        pass


class SythesisData(DataSet):
    """
       CVPR 2016 :  A large Dataset to train Convolutional networks for disparity.optical flow,and scene flow estimation

       If you want to check setting, please refer :
            http://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """

    def __init__(self, data_list , prefix=cfg.dataset.SythesisData_prefix):

        self.dirs = []
        self.data_list = data_list
        self.get_dir(prefix)

    def get_dir(self, prefix):

        # Driving
        if 'Driving' in self.data_list:
            for render_level in ('cleanpass',):
                for focallength in ('35', '15'):
                    for orient in ('forwards', 'backwards'):
                        for speed in ('fast', 'slow'):
                            img_dir = prefix + '{}/frames_{}/{}mm_focallength/scene_{}/{}/'.format('Driving',
                                                                                                   render_level,
                                                                                                   focallength,
                                                                                                   orient, speed)
                            dis_dir = prefix + '{}/disparity/{}mm_focallength/scene_{}/{}/'.format('Driving',
                                                                                                     focallength,
                                                                                                     orient, speed)

                            change_dir = prefix + '{}/disparity_change/{}mm_focallength/scene_{}/{}/{}/{}/'.format(
                                                                                                        'Driving',
                                                                                                        focallength,
                                                                                                        orient,
                                                                                                        speed,
                                                                                                        'into_future', 'left')

                            flow_dir = prefix + '{}/optical_flow/{}mm_focallength/scene_{}/{}/{}/{}/'.format(
                                                                                                             'Driving',
                                                                                                             focallength,
                                                                                                             orient,
                                                                                                             speed,
                                                                                                             'into_future','left')
                            num = len(glob.glob(img_dir + 'left/*'))
                            for i in xrange(1, num):
                                if focallength == '35' and orient == 'forwards' and (i == 264 or i == 265) and speed =='fast':
                                    continue
                                self.dirs.append((img_dir + 'left/%04d.png' % i,
                                                  img_dir + 'right/%04d.png' % i,
                                                  img_dir + 'left/%04d.png' % (i+1),
                                                  img_dir + 'right/%04d.png' % (i+1),
                                                  dis_dir + 'left/%04d.pfm' % i,
                                                  dis_dir + 'left/%04d.pfm' % (i+1),
                                                  change_dir + '%04d.pfm' % i,
                                                  flow_dir + 'OpticalFlowIntoFuture_%04d_L.pfm' % i
                                                  ))

        ## Monkaa
        if 'Monkaa' in self.data_list:
            for render_level in ('cleanpass',):  # ''):
                scenes = glob.glob(prefix + '{}/frames_{}/*'.format('Monkaa', render_level))
                for item in scenes:
                    scene = item.split('/')[-1]
                    num = len(glob.glob(prefix + '{}/frames_{}/{}/left/*'.format('Monkaa', render_level, scene)))
                    img_dir = prefix + '{}/frames_{}/{}/'.format('Monkaa', render_level, scene)
                    dis_dir = prefix + '{}/disparity/{}/'.format('Monkaa', scene)
                    change_dir = prefix + '{}/disparity_change/{}/{}/{}/'.format('Monkaa', scene, 'into_future','left')
                    flow_dir = prefix + '{}/optical_flow/{}/{}/{}/'.format('Monkaa', scene, 'into_future','left')
                    for i in xrange(0, num-1):
                        self.dirs.append((img_dir + 'left/%04d.png' % i,
                                          img_dir + 'right/%04d.png' % i,
                                          img_dir + 'left/%04d.png' % (i+1),
                                          img_dir + 'right/%04d.png' % (i+1),
                                          dis_dir + 'left/%04d.pfm' % i,
                                          dis_dir + 'left/%04d.pfm' % (i+1),
                                          change_dir + '%04d.pfm' % (i),
                                          flow_dir + 'OpticalFlowIntoFuture_%04d_L.pfm' % i
                                          ))

        # flyingthing3D
        if 'flyingthing3d' in self.data_list:
            for render_level in ['cleanpass',]:
                for style in ('TRAIN',):
                    for c in ('A', 'B', 'C'):

                        num = glob.glob(prefix + '{}/frames_{}/{}/{}/*'.format('FlyingThings3D_release',
                                                                               render_level,
                                                                               style,
                                                                               c))
                        for item in num:
                            j = item.split('/')[-1]
                            img_dir = prefix + '{}/frames_{}/{}/{}/{}/'.format('FlyingThings3D_release',
                                                                               render_level,
                                                                               style,
                                                                               c,
                                                                               j)
                            dis_dir = prefix + '{}/disparity/{}/{}/{}/'.format('FlyingThings3D_release',
                                                                                 style,
                                                                                 c,
                                                                                 j)
                            change_dir = prefix + '{}/disparity_change/{}/{}/{}/{}/left/'.format('FlyingThings3D_release',
                                                                                                style,
                                                                                                c,
                                                                                                j,
                                                                                                'into_future')

                            flow_dir = prefix + '{}/optical_flow/{}/{}/{}/{}/left/'.format('FlyingThings3D_release',
                                                                                                style,
                                                                                                c,
                                                                                                j,
                                                                                                'into_future')
                            for i in xrange(6, 15):
                                self.dirs.append((img_dir + 'left/%04d.png' % i,
                                                  img_dir + 'right/%04d.png' % i,
                                                  img_dir + 'left/%04d.png' % (i+1),
                                                  img_dir + 'right/%04d.png' % (i+1),
                                                  dis_dir + 'left/%04d.pfm' % i,
                                                  dis_dir + 'left/%04d.pfm' % (i+1),
                                                  change_dir + '%04d.pfm' % (i),
                                                  flow_dir + 'OpticalFlowIntoFuture_%04d_L.pfm' % i
                                                  ))
    @staticmethod
    def shapes():
        return 540, 960

    def name(self):
        return 'synthesisData' + '_'.join(self.data_list)

    @staticmethod
    def get_data(img_dir):

        img1 = cv2.imread(img_dir[0])
        img2 = cv2.imread(img_dir[1])
        img3 = cv2.imread(img_dir[2])
        img4 = cv2.imread(img_dir[3])

        dis1, scale = util.readPFM(img_dir[4])
        dis1 = dis1 *scale

        dis2, scale = util.readPFM(img_dir[5])
        dis2 = dis2 * scale

        change, scale = util.readPFM(img_dir[6])
        change = change * scale

        flow, scale = util.readPFM(img_dir[7])
        flow = flow[:, :, :2]
        flow = flow * scale

        return img1, img2, img3, img4, dis1, dis2, change, flow, img_dir[0].split('/')[-1]

class KittiDataset(DataSet):
    '''
        Kitti stereo 2015 dataset : http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
    '''
    def __init__(self,is_train,prefix=cfg.dataset.kitti_prefix):

        if is_train == False:
            prefix = prefix + 'testing/'

        self.dirs = []

        dis1_dir = 'disp_occ_0/'
        dis2_dir = 'disp_occ_1/'
        flow_dir = 'flow_occ_0/'
        imgl_dir = 'image_2/'
        imgr_dir = 'image_3/'

        for num in xrange(0, 200):

            img1_left = prefix + imgl_dir + '%06d_10.png' % num
            img2_left = prefix + imgl_dir + '%06d_11.png' % num

            img1_right = prefix + imgr_dir + '%06d_10.png' % num
            img2_right = prefix + imgr_dir + '%06d_11.png' % num

            dir_name = '%06d' % num

            dis1 = prefix + dis1_dir + dir_name + '_10.png'.format(num)
            dis2 = prefix + dis2_dir + dir_name + '_10.png'.format(num)
            flow = prefix + flow_dir + dir_name + '_10.png'.format(num)

            self.dirs.append((img1_left, img1_right, img2_left, img2_right, dis1, dis2, flow))

    @staticmethod
    def shapes():
        return 375, 1242

    @staticmethod
    def name():
        return 'kitti'

    @staticmethod
    def get_data(img_dir):

        img1_left  = cv2.imread(img_dir[0])
        img1_right = cv2.imread(img_dir[1])

        img2_left  = cv2.imread(img_dir[2])
        img2_right = cv2.imread(img_dir[3])

        try:
            dis1 = np.round(io.imread(img_dir[4]) / 256.0)
            dis1[dis1 < 0.0001] = np.nan
            dis2 = np.round(io.imread(img_dir[5]) / 256.0)
            flow = cv2.imread(img_dir[6], cv2.IMREAD_UNCHANGED)

            valid = flow[:, :, 0]
            flow = flow[:, :, 1:]
            flow = flow.astype(np.float64)
            flow = (flow - 2 ** 15) / 64.0
            tmp = np.zeros_like(flow)
            tmp[:, :, 0] = flow[:, :, 1]
            tmp[:, :, 1] = flow[:, :, 0]
            flow = tmp

            return img1_left, img1_right, img2_left, img2_right, dis1,dis2,flow,valid

        except :

            return img1_left, img1_right, img2_left, img2_right, None, None,None,None


