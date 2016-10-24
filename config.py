from easydict import EasyDict as edict

cfg = edict()
cfg.MODEL = edict()
cfg.ADAM = edict()
cfg.dataset = edict()

cfg.MODEL.loss1_scale = 0.65
cfg.MODEL.loss2_scale = 0.25
cfg.MODEL.loss3_scale = 0.10

cfg.MODEL.epoch_num = 200
cfg.MODEL.weight_init_scale = 8
cfg.MODEL.checkpoint_prefix = '/rawdata/check_point/joint/'

cfg.ADAM.beta1 = 0.9
cfg.ADAM.beta2 = 0.999
cfg.ADAM.epsilon = 1e-08

cfg.dataset.kitti_prefix = '/rawdata/stereo/stereo&flow/kitti/'
cfg.dataset.flyingchairs_prefix = '/rawdata/stereo/FlyingChairs_release/data/'
cfg.dataset.SythesisData_prefix = '/rawdata/stereo/'
cfg.record_prefix = '/data01/stereo_rec/'

# augmentation
cfg.dataset.augment_ratio = 0.00
cfg.dataset.rotate_range = 0.0
cfg.dataset.translation_range = 0.00
cfg.dataset.gaussian_noise = 0.01

# beta: brightness    alpha :  contrastness
cfg.dataset.beta = (-0.01, 0.01)
cfg.dataset.alpha = (0.8, 1.2)
cfg.dataset.rgbmul = (0.8, 1.3)
cfg.dataset.scale = (0.8, 1.2)

# training batch size
batchsize = (6, 3, 384, 768)


