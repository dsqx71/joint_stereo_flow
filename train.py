import argparse
import logging
import mxnet as mx
import model
import dataiter
import util
import metric
import dataset
from config import cfg, batchsize

# parse parameter
parser = argparse.ArgumentParser()
parser.add_argument('--continue', action='store', dest = 'con', type=int)
parser.add_argument('--lr', action='store', dest  = 'lr', type=float)
parser.add_argument('--model', action='store', dest = 'model', type=str,default='slow')
parser.add_argument('--gpus', type=str, help = 'the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--thread', type=int, default = 30)
cmd = parser.parse_args()

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

# load net and args
if cmd.model == 'slow':
	net = model.joint_slow()
	label_shapes = [(192, 384), (96, 192), (24, 48)]
elif cmd.model == 'fast':
	net = model.joint_fast()
	label_shapes = [(192, 384), (96, 192), (24, 48)]

if cmd.con == 0:
	# restart
	args = None
	auxs = None
else:
	# continue
	args, auxs = util.load_checkpoint(cfg.MODEL.checkpoint_prefix + 'joint_net' +'_' + cmd.model , cmd.con)
	logging.info("load the {} th epoch paramaters".format(cmd.con))

optimizer = util.Adam(learning_rate=cmd.lr, beta1 = cfg.ADAM.beta1, beta2 = cfg.ADAM.beta2,
					  epsilon = cfg.ADAM.epsilon, wd = 0.00001)
optimizer.idx2name =util.get_idx2name(net)


data = dataiter.Dataiter(dataset.SythesisData(['flyingthing3d']),
						 batch_shape = batchsize,
						 label_shapes = label_shapes,
						 augment_ratio = cfg.dataset.augment_ratio,
						 n_thread = cmd.thread,
						 be_shuffle = True)
ctx =  [mx.gpu(int(i)) for i in cmd.gpus.split(',')]
model = mx.model.FeedForward(ctx = ctx,
							 symbol=net,
							 num_epoch=cfg.MODEL.epoch_num,
							 optimizer = optimizer,
							 initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='in',
															   magnitude=cfg.MODEL.weight_init_scale),
							 arg_params  = args,
							 aux_params = auxs,
							 begin_epoch = cmd.con,
							 lr_scheduler = mx.lr_scheduler.FactorScheduler(10*20000/batchsize[0],0.9))

model.fit(X = data,
		  eval_metric=[metric.EndPointErr()],
		  epoch_end_callback=mx.callback.do_checkpoint(cfg.MODEL.checkpoint_prefix + 'joint_net' +'_' + cmd.model ),
		  batch_end_callback=[mx.callback.Speedometer(batchsize[0],10)],
		  kvstore='local_allreduce_device')
