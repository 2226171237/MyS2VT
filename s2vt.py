# -*- coding=utf-8 -*-
import tensorflow as tf
import math
import argparse
from rnn_model import CaptionGenerator
from msvd_datset import DataLoader
import os
import time
import matplotlib.pyplot as plt

SAVEPATH = 'F:\\DataSets\\VideoCaption\\MSVD\\Features'
CSV_PATH='./MSR_Video_Description_Corpus.csv'
WEIGHT_SAVE_PATH='./save_model'

parse=argparse.ArgumentParser()
parse.add_argument('--mode',default='train',help='train or predict')
parse.add_argument('--cnn',default='mobilenetv2',help='vgg16, resnet50 or mobilenetv2')
parse.add_argument('--num_epochs',type=int,default=5)
parse.add_argument('--lr',type=float,default=0.1)
parse.add_argument('--batch_size',type=int,default=32)
parse.add_argument('--data_dir',default=SAVEPATH)
parse.add_argument('--csv_path',default=CSV_PATH)

args=parse.parse_args()

dataloader=DataLoader(args.csv_path,data_dir=args.data_dir)

batch_nums=int(dataloader.num_captions//args.batch_size*args.num_epochs)

model=CaptionGenerator(n_words=dataloader.vacabs.n_words,
                       batch_size=args.batch_size,
                       dim_feature=1280,
                       dim_hidden=500,
                       n_video_lstm=80,
                       n_caption_lstm=20
                       )



def lr_schedule_exponential_decay(step,init_lr=args.lr,decay_rate=0.96,decay_steps=200):
    '''学习率指数衰减'''
    return init_lr*decay_rate**math.floor(step/decay_steps)

def lr_schedule_polynomial_decay_cycle(step,init_lr=args.lr,lr_end=1e-6,decay_steps=2000,power=1,cycle=False):
    '''学习率多项式衰减，或多项式循环衰减'''
    #decay_steps = decay_steps * ceil(step / decay_steps)
    if cycle:
        fc=1 if step==0 else math.ceil(step/decay_steps)
        decay_steps*=fc
    return (init_lr-lr_end)*(1-step/decay_steps)**power+lr_end

optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr)

lr_history=[]
loss_history=[]
loss_smooth,beta,last_loss=[],0.8,0.

start_time=time.time()

fig=plt.figure(figsize=(16,12))
for batch_idx in range(batch_nums):
    video_features,captions=dataloader.get_batch(batch_size=args.batch_size)
    captions_mask=captions>0
    with tf.GradientTape() as tape:
        loss=model(video_features,captions,captions_mask)
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

    lr_history.append(tf.keras.backend.get_value(optimizer.lr))
    tf.keras.backend.set_value(optimizer.lr,lr_schedule_polynomial_decay_cycle(batch_idx,cycle=True))

    loss_history.append(loss.numpy())
    if batch_idx==0:
        loss_smooth.append(loss.numpy())
    else:
        loss_smooth.append(last_loss * beta + (1 - beta) * loss.numpy())
    last_loss=loss_smooth[-1]

    if (batch_idx+1)%10==0:
        time_used=time.time()-start_time
        print('%d/%d: loss %f, average time cost %fs' % (batch_idx+1,batch_nums,loss.numpy(),time_used/10))
        start_time=time.time()
    # 保存前20个权重文件
    if (batch_idx+1)%200==0:
        print('%d/%d: loss %f' % (batch_idx + 1, batch_nums, loss.numpy()))
        save_path=WEIGHT_SAVE_PATH+'/model_%d.h5' % (batch_idx+1)
        model.save_weights(save_path)
        print('save model weights to %s' % save_path)
        weights_file=os.listdir(WEIGHT_SAVE_PATH)
        if (len(weights_file)>20):
            del_file=sorted(weights_file)[0]
            os.remove(os.path.join(WEIGHT_SAVE_PATH,del_file))

        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.plot(loss_history,'g',alpha=0.7,linewidth=0.5)
        ax.plot(loss_smooth,'r')
        ax2.plot(lr_history,'b')
        ax.set_xlabel('batch')
        ax.set_ylabel('loss')
        ax.legend(['loss','smooth loss'])
        ax2.legend(['learningrate'])
        plt.savefig('loss.png')
        fig.clf()

print('train over')

