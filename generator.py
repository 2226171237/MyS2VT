#-*- coding:utf-8 -*-

from msvd_datset import MSVD_Caption,Vocabs,replace_map
from extractRGBfeatures import preprocess_image
from cnn_model import CNN
from rnn_model import CaptionGenerator
import numpy as np
import argparse
import cv2


def main(arg):
    # build vocab
    CSV_PATH = './MSR_Video_Description_Corpus.csv'
    data=MSVD_Caption(CSV_PATH)
    print(data.head())
    captions = data['Description'].values
    del data
    vocabs=Vocabs(list(replace_map(captions)))
    # build model
    cnn=CNN(arg.net) if arg.net else CNN()
    cnn.load_weights(arg.cnn_weight_path)
    rnn=CaptionGenerator(n_words=vocabs.n_words,
                         batch_size=1,
                         dim_feature=1280,
                         dim_hidden=500,
                         n_video_lstm=80,
                         n_caption_lstm=20,
                         bias_init_vector=vocabs.bias_init_vector)
    rnn.load_weights(arg.rnn_weight_path,by_name=True)
    # extract video features
    print('extract %s video features' % (arg.video_path))
    this_features = []
    if arg.video_path.endswith('.avi'):
        cap = cv2.VideoCapture(arg.video_path)
        flame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频总帧数
        if flame_count > rnn.n_video_lstm:  # 高于要求的帧数均匀采样
            select_flames = np.linspace(0, flame_count, num=rnn.n_video_lstm, dtype=np.int32)
        else:
            select_flames = np.arange(0, flame_count, dtype=np.int32)
        print('flame count:', flame_count, 'select: ', select_flames[:10])
        flames = []
        i, flame_index, selected_num = 0, 0, 0
        while True:
            ret, flame = cap.read()
            if ret is False:
                break
            if i == select_flames[flame_index]:
                flame_index += 1
                selected_num += 1
                flame = preprocess_image(flame)
                flames.append(flame)
            if selected_num == 64:
                selected_num = 0
                this_features.append(cnn.get_features(np.array(flames)))
                flames = []
            i += 1
            if i == flame_count and flames:
                this_features.append(cnn.get_features(np.array(flames)))
    else:
        raise ValueError("only support .avi video")
    #  generator caption
    this_features=np.concatenate(this_features,axis=0)
    this_feature_nums, dims_feature = this_features.shape
    if this_feature_nums < rnn.n_video_lstm:  # 小于需要的帧数则填充0
        this_features = np.vstack([this_features,
                                   np.zeros(shape=(rnn.n_video_lstm - this_feature_nums, dims_feature))])
    if this_feature_nums > rnn.n_video_lstm:  # 大于指定帧数则使用均匀采样
        selected_idxs = np.linspace(0, this_feature_nums, num=rnn.n_video_lstm)
        this_features = this_features[selected_idxs, :]
    generator=rnn.predict(this_features.reshape(1,*this_features.shape))
    captions=[]
    for i, gen_caption in enumerate(generator):
        sent = []
        for ii in gen_caption:
            if ii == vocabs.word2idx['<eos>']:
                break
            if ii == vocabs.word2idx['<pad>']:
                continue
            if ii != vocabs.word2idx['<bos>']:
                sent.append(vocabs.idx2word[ii])

        caption = ' '.join(sent)
        captions.append(caption)
    return captions


if __name__=="__main__":
    parse=argparse.ArgumentParser('generator')
    parse.add_argument('--net',type=str,default=None,help='vgg16 ,resnet50 or mobilenetv2(defalut)')
    parse.add_argument('--cnn_weight_path',type=str,default='./cnn_weights/mobilenet_v2_notop.h5',help='the cnn model pretrained weight path')
    parse.add_argument('--rnn_weight_path',type=str,default='./save_model/model_600.h5',help='the rnn model pretrained weight path')
    parse.add_argument('--video_path',type=str,default=None,help='the path of your video')

    arg=parse.parse_args()
    captions=main(arg)
    for cap in captions:
        print("caption: ",cap)




