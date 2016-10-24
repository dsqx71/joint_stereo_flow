import mxnet as mx

def get_conv(name, data, num_filter, kernel, stride, pad,
             with_relu, bn_momentum, dilate=(1, 1)):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilate=dilate,
        no_bias=True
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        eps=1e-5 + 1e-10 # issue of cudnn
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.LeakyReLU(name=name + '_prelu', act_type='prelu', data=bn)
        if with_relu else bn
    )


def initila_block(data):
    # TODO: input shape: (1, 3, 1086, 2173) shape incorrect
    conv = mx.symbol.Convolution(
        name="initial_conv",
        data=data,
        num_filter=13,
        kernel=(3, 3),
        stride=(2, 2),
        pad=(1, 1),
        no_bias=True
    )

    maxpool = mx.symbol.Pooling(data=data, pool_type="max", kernel=(2, 2), stride=(2, 2),
                                name="initial_maxpool")
    concat = mx.symbol.Concat(
        conv,
        maxpool,
        num_args=2,
        name="initial_concat"
    )
    return concat


def make_block(name, data, num_filter, bn_momentum,
               down_sample=False, up_sample=False,
               dilated=(1, 1), asymmetric=0):
    """maxpooling & padding"""
    if down_sample:
        # 1x1 conv ensures that channel equal to main branch
        maxpool = get_conv(name=name + '_proj_maxpool',
                           data=data,
                           num_filter=num_filter,
                           kernel=(2, 2),
                           pad=(0, 0),
                           with_relu=True,
                           bn_momentum=bn_momentum,
                           stride=(2, 2))

    elif up_sample:
        # maxunpooling.
        maxpool = mx.symbol.Deconvolution(name=name + '_unpooling',
                                   data=data,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))

        # Reference: https://github.com/e-lab/ENet-training/blob/master/train/models/decoder.lua
        # Padding is replaced by 1x1 convolution
        maxpool = get_conv(name=name + '_padding',
                           data=maxpool,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=(1, 1),
                           pad=(0, 0),
                           bn_momentum=bn_momentum,
                           with_relu=False)
    # main branch begin
    proj = get_conv(name=name + '_proj0',
                    data=data,
                    num_filter=num_filter,
                    kernel=(1, 1) if not down_sample else (2, 2),
                    stride=(1, 1) if not down_sample else (2, 2),
                    pad=(0, 0),
                    with_relu=True,
                    bn_momentum=bn_momentum)

    if up_sample:
        conv = mx.symbol.Deconvolution(name=name + '_deconv',
                                   data=proj,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))
    else:
        if asymmetric == 0:
            conv = get_conv(name=name + '_conv',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(3, 3),
                            pad=dilated,
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
        else:
            conv = get_conv(name=name + '_conv1',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(1, asymmetric),
                            pad=(0, asymmetric / 2),
                            stride=(1, 1),
                            dilate=dilated,
                            with_relu=True,
                            bn_momentum=bn_momentum)
            conv = get_conv(name=name + '_conv2',
                            data=conv,
                            num_filter=num_filter,
                            kernel=(asymmetric, 1),
                            pad=(asymmetric / 2, 0),
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)

    regular = mx.symbol.Convolution(name=name + '_expansion',
                                        data=conv,
                                        num_filter=num_filter,
                                        kernel=(1, 1),
                                        pad=(0, 0),
                                        stride=(1, 1),
                                        no_bias=True)
    regular = mx.symbol.BatchNorm(
        name=name + '_expansion_bn',
        data=regular,
        fix_gamma=False,
        momentum=bn_momentum,
        eps=1e-5 + 1e-10 # issue of cudnn
    )
    # main branch end
    # TODO: spatial dropout

    if down_sample or up_sample:
        regular = mx.symbol.ElementWiseSum(maxpool, regular, name =  name + "_plus")
    else:
        regular = mx.symbol.ElementWiseSum(data, regular, name =  name + "_plus")
    regular = mx.symbol.LeakyReLU(name=name + '_expansion_prelu', act_type='prelu', data=regular)
    return regular


def get_body(data, num_class, bn_momentum):
    ##level 0
    data = initila_block(data)  # 16

    ##level 1
    num_filter = 64
    data = data0 = make_block(name="bottleneck1.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)  # 64
    for block in range(4):
        data = make_block(name='bottleneck1.%d' % (block + 1),
                          data=data, num_filter=num_filter,  bn_momentum=bn_momentum,
                          down_sample=False, up_sample=False)
    data0 = make_block(name="projection1", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    ##level 2
    num_filter = 128
    data = data0 = make_block(name="bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))

    data = make_block(name="bottleneck2.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck2.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection2", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    ##level 3
    num_filter = 128
    data = data0 = make_block(name="bottleneck3.1", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum)
    data = make_block(name="bottleneck3.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck3.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck3.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck3.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck3.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck3.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck3.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))
    data = make_block(name="bottleneck3.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck3.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection3", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    ##level 4
    num_filter = 64
    data = data0 = make_block(name="bottleneck4.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = make_block(name="bottleneck4.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = make_block(name="bottleneck4.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)

    # ##level 5
    num_filter = 16
    data = data0 = make_block(name="bottleneck5.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = make_block(name="bottleneck5.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)

    data = mx.symbol.Deconvolution(data=data, kernel=(16, 16), stride=(2, 2), num_filter=num_class,
                                   name="fullconv")  # mx.symbol.UpSampling(data,scale=2,sample_type='nearest',num_args=1,name="fullconv")#

    return data


def get_symbol(num_classes, bn_momentum=0.9, spl=0, pool_kernel=(8, 8)):
    data = mx.symbol.Variable(name='data')
    # shape = mx.symbol.Variable(name='shape')
    # Simulate z-score normalization as that in
    # https://github.com/gcr/torch-residual-networks/blob/master/data/cifar-dataset.lua
    # zscore = mx.symbol.BatchNorm(
    #     name='zscore',
    #     data=data,
    #     fix_gamma=True,
    #     momentum=bn_momentum
    # )
    # conv = get_conv(
    #     name='conv0',
    #     data=zscore,
    #     num_filter=num_filter,
    #     kernel=(3, 3),
    #     stride=(1, 1),
    #     pad=(1, 1),
    #     with_relu=True,
    #     bn_momentum=bn_momentum
    # )
    body = get_body(
        data,
        num_classes,
        bn_momentum
    )

    body = mx.symbol.Crop(*[body, data], name="fullconv_crop")
    if spl == 0:
        softmax = mx.symbol.SoftmaxOutput(data=body, multi_output=True, use_ignore=True, ignore_label=255,
                                          name="softmax")  # is_hidden_layer=True,
    else:
        softmax = mx.symbol.SoftmaxOutput(data=body, multi_output=True, use_ignore=True, ignore_label=255,
                                          is_hidden_layer=True,
                                          name="softmax")  # is_hidden_layer=True,
    # hiddensoftmax = mx.symbol.SoftmaxOutput(data=body, multi_output=True, use_ignore=True, is_hidden_layer=True, name ='hiddensoftmax')
    # weighted_softmax = SPSoftmaxOutput()
    # softmax = weighted_softmax(data=hiddensoftmax,name="softmax")
    # return mx.symbol.Group([softmax, body])  # mx.symbol.SoftmaxOutput(data=body, name='softmax')
    return softmax

