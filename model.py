import mxnet as mx
from config import cfg


class SparseRegressionLoss(mx.operator.NumpyOp):
    '''
        if label is nan, don't compute gradient
    '''

    def __init__(self,is_l1,loss_scale):

        super(SparseRegressionLoss, self).__init__(False)
        self.is_L1 = is_l1
        self.loss_scale = loss_scale

    def list_arguments(self):

        return ['data', 'label']

    def list_outputs(self):

        return ['output']

    def infer_shape(self, in_shape):

        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]

        return [data_shape, label_shape], [output_shape]

    def forward(self,in_data,out_data):

        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self,out_grad, in_data, out_data, in_grad):

        label = in_data[1]
        y = out_data[0]
        mask = (label!=label)
        label[mask] = y[mask]
        # mask = (np.abs(y-label) >= 1) & (np.abs(y-label)/np.abs(label)>=0.02)
        dx = in_grad[0]
        if self.is_L1:
            dx[:] = np.sign(y-label)*self.loss_scale
        else:
            dx[:] = (y-label)*self.loss_scale


def block(data,num_filter,name,is_downsample=False,return_regress=False):

    # param = num_filter ^2 * 6 * 3 * 3

    stride = (2,2) if is_downsample else (1,1)
    data1 = get_conv(name+'block.0', data, num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), dilate=(1, 1), bn=True)

    regressor = get_conv(name + 'regressor.0', data, num_filter=5, kernel=(5, 5),
                         stride=stride, pad=(2, 2), dilate=(1, 1), no_bias=False, is_conv=True, with_relu=False)
    tmp = data = mx.sym.Concat(data1,regressor)

    tmp2 = data = get_conv(name+'block.1', data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True)

    data = mx.sym.Concat(tmp, data)
    data = get_conv(name+'block.2', data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True)

    data = mx.sym.Concat(data, tmp2)
    data = get_conv(name+'block.3', data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True)

    regressor = get_conv(name + 'regressor.1', data, num_filter=5, kernel=(5, 5),
                         stride=(1, 1), pad=(2, 2), dilate=(1, 1), no_bias=False, is_conv=True, with_relu=False)

    data = mx.sym.Concat(data, regressor)

    if return_regress:
        return data,regressor
    else:
        return data

def correlation_unit(data1,data2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply,name,data_type):

    if data_type == 'stereo':
        corr = mx.sym.Correlation1D(data1=data1, data2=data2, pad_size=pad_size, kernel_size=kernel_size,
                                     max_displacement=max_displacement, stride1=stride1, stride2=stride2)
    else:
        corr = mx.sym.Correlation(data1=data1, data2=data2, kernel_size=kernel_size, max_displacement=max_displacement,
                                  stride1=stride1, stride2=stride2, pad_size=pad_size, is_multiply=is_multiply,
                                  name=name)
    return corr

def get_conv(name, data, num_filter, kernel, stride, pad, dilate=(1, 1), no_bias=False, with_relu=True, weight=None, is_conv=True,bn=False):

    if is_conv:
        if weight is None:
            conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                         pad=pad, dilate=dilate, no_bias=no_bias, workspace=1024)
        else:
            conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                         pad=pad, dilate=dilate, no_bias=no_bias, weight=weight, workspace=1024)
    else:
        conv = mx.sym.Deconvolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                    no_bias=no_bias)
    if bn:
        conv = mx.symbol.BatchNorm( name=name + '_bn', data=conv, fix_gamma=False, momentum=0.90, eps=1e-5 + 1e-10)
    return mx.sym.LeakyReLU(name=name + '_prelu', data=conv, act_type='prelu') if with_relu else conv

def get_feature(data,name):

    num_filter = 32
    tmp = data = get_conv('level0.0' + name, data, num_filter,
                    kernel=(5, 5), stride=(2, 2), pad=(2, 2), dilate=(1, 1), no_bias=False, bn=True)
    data = get_conv('level0.1' + name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False, bn=True)
    data0 = data = data + tmp

    num_filter = 64
    tmp = data = get_conv('level1.0' + name, data, num_filter,
                    kernel=(3, 3), stride=(2, 2), pad=(1, 1), dilate=(1, 1), no_bias=False, bn=True)
    data = get_conv('level1.1' + name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False,  bn=True)
    data = mx.sym.Concat(tmp,data)
    data = get_conv('level1.2' + name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False, bn=True)
    data1 = data

    return data0, data1

def upscale(data,num_filter,name):

    data = get_conv(name + 'deconv', data, num_filter=num_filter, pad=(1, 1), kernel=(4, 4), stride=(2, 2), is_conv=False, with_relu=False)
    return data

# def get_loss(data,label,grad_scale,name,get_data=False):
#
#     loss = mx.symbol.CaffeLoss(data=data, label=label,
#                                name=name,
#                                prototxt='''
#                                layer {
#                                   type: "L1Loss"
#                                   loss_weight: %f
#                                   l1_loss_param {
#                                     l2_per_location: false
#                                     normalize_by_num_entries: true
#                                   }
#                                 }'''  % grad_scale)
#
#     return (loss,data) if get_data else loss

def get_loss(data,label,grad_scale,name,get_data=False, is_sparse = False):

    # data = mx.sym.Activation(data=data, act_type='relu',name=name+'relu')

    if  is_sparse:
        loss = SparseRegressionLoss(is_l1=False, loss_scale=grad_scale)
        loss = loss(data=data, label=label)
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=grad_scale)

    return (loss,data) if get_data else loss

def joint_slow(loss1_scale=cfg.MODEL.loss1_scale, loss2_scale=cfg.MODEL.loss2_scale,
               loss3_scale=cfg.MODEL.loss3_scale):

    img1left = mx.sym.Variable('img1left_data')
    img1right = mx.sym.Variable('img1right_data')

    img2left = mx.sym.Variable('img2left_data')
    img2right = mx.sym.Variable('img2right_data')

    label1 = mx.sym.Variable('label1')
    label2 = mx.sym.Variable('label2')
    label3 = mx.sym.Variable('label3')

    data0_l1, data1_l1 = get_feature(img1left, 'img1left')
    data0_l2, data1_l2 = get_feature(img2left, 'img2left')
    data0_r1, data1_r1 = get_feature(img1right, 'img1right')
    data0_r2, data1_r2 = get_feature(img2right, 'img2right')

    all_img = mx.sym.Concat(data0_l1,data0_l2,data0_r1,data0_r2)
    all_img = block(data = all_img, num_filter = 64,   name='all_img1', is_downsample=True)
    all_img = block(data = all_img, num_filter = 128,  name='all_img2', is_downsample=True)

    # 32 x 96 x 192
    corr0_left1_right1 = correlation_unit(data0_l1, data0_r1, kernel_size=1, max_displacement=16, stride1=1,
                                          stride2=1, pad_size=16, is_multiply=True, name='level1_corrl1_r1',
                                          data_type='stereo')

    corr0_left2_right2 = correlation_unit(data0_l2, data0_r2, kernel_size=1, max_displacement=16, stride1=1,
                                          stride2=1, pad_size=16, is_multiply=True, name='level1_corrl2_r2',
                                          data_type='stereo')

    corr0_left1_left2  = correlation_unit(data0_l1, data0_l2, kernel_size=1, max_displacement= 8, stride1=1,
                                          stride2=2, pad_size=8, is_multiply=True, name='level1_corrl1_l2',
                                          data_type='flow')

    tmp = mx.sym.Concat(corr0_left1_right1, corr0_left2_right2, corr0_left1_left2)
    downsample = block(data = tmp, num_filter = 64,     name = 'downsample1', is_downsample=True)
    downsample = block(data = downsample, num_filter=128, name='downsample2', is_downsample=True)

    corr1_left1_right1 = correlation_unit(data1_l1, data1_r1, kernel_size = 1, max_displacement = 64, stride1 = 1,
                                         stride2 = 1, pad_size = 64 , is_multiply = True, name = 'level2_corrl1_r1',
                                         data_type='stereo')

    corr1_left2_right2 = correlation_unit(data1_l2, data1_r2, kernel_size = 1, max_displacement = 64, stride1 = 1,
                                         stride2 = 1, pad_size = 64 , is_multiply = True, name = 'level2_corrl2_r2',
                                         data_type = 'stereo')

    corr1_left1_left2 = correlation_unit(data1_l1, data1_l2, kernel_size=1, max_displacement= 64, stride1 = 2,
                                         stride2 = 2, pad_size = 64, is_multiply=True, name= 'level3_corrl1_l2',
                                         data_type='flow')

    data1_l1_r1 = block(data = corr1_left1_right1, num_filter=64, name='l1_r1_reduce')
    data1_l2_r2 = block(data = corr1_left2_right2, num_filter=64, name='l2_r2_reduce')
    data1_l1_l2 = block(data = corr1_left1_left2, num_filter=128, name='l1_l1_reduce')

    tmp2 = data = mx.sym.Concat(data1_l1_r1, data1_l2_r2)
    data = get_conv('level3_conv', data, 192, kernel=(3, 3), stride=(2, 2), pad=(1, 1), dilate=(1, 1), bn=True, with_relu=False)
    data = mx.sym.Concat(data, data1_l1_l2, downsample, all_img)
    tmp1 = data = block(data=data, num_filter=192, name='level3')

    data = block(data=data, num_filter=256, name='level4', is_downsample=True)

    data = block(data=data, num_filter=512, name='level5a', is_downsample=True)
    data = block(data=data, num_filter=512, name='level5b')

    data = upscale(data=data, num_filter=256, name='level6a_up')
    data, pr2 = block(data=data,   num_filter=192, name='level6a',return_regress=True)

    data = upscale(data=data, num_filter=128, name='level6b_up')
    data = mx.sym.Concat(data,tmp1)
    data = block(data=data, num_filter=96, name='level6b')

    data = upscale(data=data, num_filter=64, name='level7_up')
    data = mx.sym.Concat(data, tmp2)
    data, pr1 = block(data=data, num_filter=48, name='level7', return_regress=True)

    data = upscale(data=data, num_filter = 32, name='level8_up')
    data = mx.sym.Concat(data0_l1,data0_l1,data)
    data = get_conv('level8_conv', data, 32, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=False)
    pr0 = mx.sym.Convolution(data, kernel=(3, 3), num_filter = 4, stride=(1, 1), pad=(1, 1), name='pr')

    loss1 = get_loss(pr0, label1, loss1_scale, name='loss1')
    loss2 = get_loss(pr1, label2, loss2_scale, name='loss2')
    loss3 = get_loss(pr2, label3, loss3_scale, name='loss3')

    # img1 = mx.sym.BlockGrad(data=img1left, name='img1')
    # img2 = mx.sym.BlockGrad(data=img1right, name='img2')
    # img3 = mx.sym.BlockGrad(data=img2left, name='img3')
    # img4 = mx.sym.BlockGrad(data=img2right, name='img4')
    #
    # label =  mx.sym.BlockGrad(data=label1, name='label')
    #
    # pr = mx.sym.BlockGrad(data=pr0, name='pr')
    # net = mx.sym.Group([loss1, loss2, loss3, img1, img2, img3, img4, label,pr])
    net = mx.sym.Group([loss1, loss2, loss3])

    return net


def joint_fast(loss1_scale=cfg.MODEL.loss1_scale, loss2_scale=cfg.MODEL.loss2_scale, loss3_scale=cfg.MODEL.loss3_scale):

    img1left = mx.sym.Variable('img1left_data')
    img1right = mx.sym.Variable('img1right_data')

    img2left = mx.sym.Variable('img2left_data')
    img2right = mx.sym.Variable('img2right_data')

    label1 = mx.sym.Variable('label1')
    label2 = mx.sym.Variable('label2')
    label3 = mx.sym.Variable('label3')

    data0_l1, data1_l1 = get_feature(img1left, 'img1left')
    data0_l2, data1_l2 = get_feature(img2left, 'img2left')
    data0_r1, data1_r1 = get_feature(img1right, 'img1right')
    data0_r2, data1_r2 = get_feature(img2right, 'img2right')

    all_img = mx.sym.Concat(data0_l1, data0_l2, data0_r1, data0_r2)
    all_img = block(data=all_img, num_filter = 48,  name='all_img', is_downsample=True)

    # 32 x 96 x 192
    corr0_left1_right1 = correlation_unit(data0_l1, data0_r1, kernel_size=1, max_displacement=20, stride1=1,
                                          stride2=1, pad_size=20, is_multiply=True, name='level1_corrl1_r1',
                                          data_type='stereo')

    corr0_left2_right2 = correlation_unit(data0_l2, data0_r2, kernel_size=1, max_displacement=20, stride1=1,
                                          stride2=1, pad_size=20, is_multiply=True, name='level1_corrl2_r2',
                                          data_type='stereo')

    corr0_left1_left2 = correlation_unit(data0_l1, data0_l2, kernel_size=1, max_displacement=8, stride1=1,
                                         stride2=2, pad_size=8, is_multiply=True, name='level1_corrl1_l2',
                                         data_type='flow')

    tmp = mx.sym.Concat(corr0_left1_right1, corr0_left2_right2, corr0_left1_left2)
    downsample = block(data=tmp, num_filter = 48, name='downsample', is_downsample=True)

    corr1_left1_right1 = correlation_unit(data1_l1, data1_r1, kernel_size=1, max_displacement=80, stride1=1,
                                          stride2=1, pad_size=80, is_multiply=True, name='level2_corrl1_r1',
                                          data_type='stereo')

    corr1_left2_right2 = correlation_unit(data1_l2, data1_r2, kernel_size=1, max_displacement=80, stride1=1,
                                          stride2=1, pad_size=80, is_multiply=True, name='level2_corrl2_r2',
                                          data_type='stereo')

    corr1_left1_left2 = correlation_unit(data1_l1, data1_l2, kernel_size=1, max_displacement=40, stride1=2,
                                         stride2=2, pad_size=40, is_multiply=True, name='level3_corrl1_l2',
                                         data_type='flow')

    data1_l1_r1 = block(data=corr1_left1_right1, num_filter=64, name='l1_r1_reduce')
    data1_l2_r2 = block(data=corr1_left2_right2, num_filter=64, name='l2_r2_reduce')
    data1_l1_l2 = block(data=corr1_left1_left2,  num_filter=128, name='l1_l1_reduce')

    data = mx.sym.Concat(data1_l1_r1, data1_l2_r2, downsample, all_img)
    tmp2 = get_conv('redir0', data,  64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True, with_relu=False)
    data = get_conv('level3_conv', data, 128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), dilate=(1, 1), bn=True, with_relu=False)
    data = mx.sym.Concat(data, data1_l1_l2)
    data = block(data=data, num_filter=128, name='level3')
    tmp1 = get_conv('redir1', data,  64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True, with_relu=False)

    data = block(data=data, num_filter=224, name='level4a', is_downsample=True)
    tmp0 = get_conv('redir2', data, 128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True, with_relu=False)

    data = block(data=data, num_filter=448, name='level5a', is_downsample=True)
    data = block(data=data, num_filter=448, name='level5b')

    data = upscale(data=data, num_filter=224, name='level6a_up')
    data = mx.sym.Concat(data,tmp0)
    data, pr2 = block(data=data, num_filter=192, name='level6a', return_regress=True)

    data = upscale(data=data, num_filter=128, name='level6b_up')
    data = mx.sym.Concat(data, tmp1)
    data = block(data=data, num_filter=96, name='level6b')

    data = upscale(data=data, num_filter=64, name='level7_up')
    data = mx.sym.Concat(data, tmp2)
    data, pr1 = block(data=data, num_filter=48, name='level7', return_regress=True)

    data = upscale(data=data, num_filter=32, name='level8_up')
    data = mx.sym.Concat(data0_l1, data0_l1, data)
    data = get_conv('level8_conv', data, 32, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=False)
    pr0 = mx.sym.Convolution(data, kernel=(3, 3), num_filter=5, stride=(1, 1), pad=(1, 1), name='pr')

    loss1 = get_loss(pr0, label1, loss1_scale, name='loss1')
    loss2 = get_loss(pr1, label2, loss2_scale, name='loss2')
    loss3 = get_loss(pr2, label3, loss3_scale, name='loss3')

    img1 = mx.sym.BlockGrad(data=img1left, name='img1')
    img2 = mx.sym.BlockGrad(data=img1right, name='img2')
    img3 = mx.sym.BlockGrad(data=img2left, name='img3')
    img4 = mx.sym.BlockGrad(data=img2right, name='img4')

    label =  mx.sym.BlockGrad(data=label1, name='label')

    pr = mx.sym.BlockGrad(data=pr0, name='pr')
    net = mx.sym.Group([loss1, loss2, loss3, img1, img2, img3, img4, label])

    # net = mx.sym.Group([loss1, loss2, loss3])

    return net









