# -*- coding=utf-8 -*-
import numpy as np
from cnn_model import CNN
import os
import cv2


VIDEOPATH='F:\\DataSets\\VideoCaption\\MSVD\\YouTubeClips'
SAVEPATH='F:\\DataSets\\VideoCaption\\MSVD\\Features'


def preprocess_image(img,target_size=(224,224)):
    if isinstance(target_size,(list,tuple)):
        if len(target_size)==1:
            target_size=(target_size[0],target_size[0])
    else:
        target_size=(target_size,target_size)
    t_height,t_weight=target_size
    img_h, img_w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    background = np.full(shape=(t_height,t_weight,3), fill_value=255, dtype=np.uint8)
    if img_h>img_w:
        w=int(img_w/img_h*t_height)
        img = cv2.resize(img, (w,t_height))
        pad_x=(t_height-w)//2
        background[:,pad_x:pad_x+w]=np.array(img,np.uint8)  # center of image
    else:
        h=int(img_h / img_w * t_weight)
        img = cv2.resize(img, (t_weight,h))
        pad_y=(t_weight-h)//2
        background[pad_y:pad_y+h,:]=np.array(img,np.uint8)
    return background.astype(np.float32)/127.5-1.0

def extractRGBfeature(video_path=VIDEOPATH,save_path=SAVEPATH,net=None,batch_size=64,num_flames=80):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cnn=CNN(net) if net else CNN()
    cnn.load_weights()
    videos=os.listdir(video_path)
    for i,video in enumerate(videos):
        print('%d/%d: eatract %s video features' % (i,len(videos),video))
        this_features=[]
        if video.endswith('.avi'):
            v_path=os.path.join(video_path,video)
            cap=cv2.VideoCapture(v_path)
            flame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT) # 视频总帧数
            if flame_count>num_flames: # 高于要求的帧数均匀采样
                select_flames=np.linspace(0,flame_count,num=num_flames,dtype=np.int32)
            else:
                select_flames=np.arange(0,flame_count,dtype=np.int32)
            print('flame count:', flame_count,'select: ',select_flames[:10])
            flames=[]
            i,flame_index,selected_num=0,0,0
            while True:
                ret,flame=cap.read()
                if ret is False:
                    break
                if i==select_flames[flame_index]:
                    flame_index+=1
                    selected_num+=1
                    flame=preprocess_image(flame)
                    flames.append(flame)
                if selected_num==batch_size:
                    selected_num=0
                    this_features.append(cnn.get_features(np.array(flames)))
                    flames=[]
                i+=1
                if i==flame_count and flames:
                    this_features.append(cnn.get_features(np.array(flames)))
            this_save_path=os.path.join(save_path,video.split('.')[0]+'.npy')
            print('save feature to :',this_save_path)
            np.save(this_save_path,np.array(this_features))
    print('eatract all video features ok!')

if __name__ == '__main__':
    extractRGBfeature()



