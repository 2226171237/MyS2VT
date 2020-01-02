# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
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
parse.add_argument('--num_epochs',type=int,default=1)
parse.add_argument('--lr',type=float,default=1e-4)
parse.add_argument('--batch_size',type=int,default=32)
parse.add_argument('--data_dir',default=SAVEPATH)
parse.add_argument('--csv_path',default=CSV_PATH)

args=parse.parse_args()

dataloader=DataLoader(args.csv_path,data_dir=args.data_dir)

batch_nums=int(dataloader.num_captions//args.batch_size*args.num_epochs)

model=CaptionGenerator(n_words=dataloader.vacabs.n_words,
                       batch_size=args.batch_size,
                       dim_feature=1280,
                       dim_hidden=100,
                       n_video_lstm=80,
                       n_caption_lstm=20
                       )

optimizer=tf.keras.optimizers.Adam(lr=args.lr)
loss_history=[]
loss_smooth,beta,last_loss=[],0.8,0.
start_time=time.time()
for batch_idx in range(batch_nums):
    video_features,captions=dataloader.get_batch(batch_size=args.batch_size)
    captions_mask=captions>0
    with tf.GradientTape() as tape:
        loss=model(video_features,captions,captions_mask)
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

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
    if (batch_idx+1)%50==0:
        print('%d/%d: loss %f' % (batch_idx + 1, batch_nums, loss.numpy()))
        save_path=WEIGHT_SAVE_PATH+'/model_%d.h5' % (batch_idx+1)
        model.save_weights(save_path)
        print('save model weights to %s' % save_path)
        plt.plot(loss_history,'g',alpha=0.7,linewidth=0.5)
        plt.plot(loss_smooth,'r')
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.legend(['loss','smooth loss'])
        plt.savefig('loss.png')
        plt.clf()
print('train over')

