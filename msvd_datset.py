# -*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import os

CSV_PATH='./MSR_Video_Description_Corpus.csv'
VIDEO_PATH='F:\\DataSets\\VideoCaptions\\MSVD\\YouTubeClips'


def replace_map(str_array):
    """字符串替换"""
    str_array = map(lambda x: x.replace('.', ''), str_array)
    str_array = map(lambda x: x.replace('/', ''), str_array)
    str_array = map(lambda x: x.replace('?', ''), str_array)
    str_array = map(lambda x: x.replace(',', ''), str_array)
    str_array = map(lambda x: x.replace(':', ''), str_array)
    str_array = map(lambda x: x.replace('"', ''), str_array)
    str_array = map(lambda x: x.replace('!', ''), str_array)
    str_array = map(lambda x: x.replace('\\', ''), str_array)
    str_array = map(lambda x: x.replace('\n', ''), str_array)
    return str_array

def MSVD_Caption(filename):
    MSVDCaption=pd.read_csv(filename)
    MSVDCaption=MSVDCaption[MSVDCaption['Language']=='English']
    MSVDCaption['Video_path']=MSVDCaption.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi',axis=1)
    MSVDCaption.dropna(axis=1,inplace=True,how='all')
    cols=list(MSVDCaption.columns)
    cols.append('CaptionNums')
    MSVDCaption=pd.merge(MSVDCaption,MSVDCaption.groupby('VideoID').count().iloc[:,0],on='VideoID',how='left')
    MSVDCaption.columns=pd.Index(cols)
    MSVDCaption.dropna(axis=0,inplace=True)
    MSVDCaption['Video_path']=MSVDCaption['Video_path'].apply(lambda x: os.path.join(VIDEO_PATH,x))
    MSVDCaption['Have']=MSVDCaption['Video_path'].apply(lambda x: os.path.exists(x))
    MSVDCaption=MSVDCaption[MSVDCaption['Have']]
    MSVDCaption=MSVDCaption[MSVDCaption['Description'].apply(lambda x : isinstance(x,str))]
    return MSVDCaption

class Vocabs:
    def __init__(self,captions, min_count_threshold=5):
        self.min_count_threshold = min_count_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.n_words = 0
        self.all_words = []
        self.count_per_word = {}
        self.bias_init_vector=None
        self._create_vcab(captions)

    def _create_vcab(self,docs):
        num_captions=0
        for caption in docs:
            num_captions+=1
            for w in caption.lower().split(' '):
                self.count_per_word[w] = self.count_per_word.get(w, 0) + 1

        n=len(self.count_per_word)
        self.all_words=[w for w in self.count_per_word if self.count_per_word[w]>self.min_count_threshold]
        self.all_words=sorted(self.all_words)

        print('filter %d words from %d words' % (len(self.all_words),n))

        self.word2idx['<pad>']=0
        self.word2idx['<bos>']=1
        self.word2idx['<eos>']=2
        self.word2idx['<unk>']=3

        self.count_per_word['<pad>']=num_captions
        self.count_per_word['<bos>']=num_captions
        self.count_per_word['<eos>']=num_captions
        self.count_per_word['<unk>']=num_captions

        for i,w in enumerate(self.all_words):
            self.word2idx[w]=i+4

        self.idx2word={self.word2idx[w]:w for w in self.word2idx}
        self.n_words=len(self.idx2word)

        self.bias_init_vector=np.array([self.count_per_word[w] for w in self.word2idx],dtype=np.float32)
        self.bias_init_vector/=np.sum(self.bias_init_vector)
        self.bias_init_vector=np.log(self.bias_init_vector)
        self.bias_init_vector-=np.max(self.bias_init_vector)


class DataLoader:
    #cache={}
    def __init__(self,csv_dir,data_dir,n_flames_per_video=80,n_words_per_caption=20):
        self.dataset=MSVD_Caption(csv_dir)
        self.video_paths=self.dataset['Video_path'].values
        self.captions=self.dataset['Description'].values
        self.captions=list(replace_map(self.captions))
        self.vacabs=Vocabs(self.captions)
        self.num_videos=len(set(self.video_paths))
        self.num_captions=len(self.video_paths)
        self.data_dir=data_dir
        self.n_flames_per_video=n_flames_per_video
        self.n_words_per_caption=n_words_per_caption

    def get_batch(self,batch_size):
        '''训练的时候使用'''
        return_features=[]
        return_captions=[]
        indices=np.random.randint(0,self.num_captions,size=batch_size)
        for idx in indices:
            this_path=os.path.basename(self.video_paths[idx]).split('.')[0]
            feature_path=os.path.join(self.data_dir,this_path+'.npy')
            if os.path.exists(feature_path):
                this_features=np.load(feature_path,allow_pickle=True)
            else:
                raise ValueError('feature path not existed in class %s' % self.__class__.__name__)
            this_features=np.vstack(this_features)
            this_feature_nums,dims_feature=this_features.shape
            if this_feature_nums<self.n_flames_per_video:  # 小于需要的帧数则填充0
                this_features=np.vstack([this_features,
                                         np.zeros(shape=(self.n_flames_per_video-this_feature_nums,dims_feature))])
            if this_feature_nums>self.n_flames_per_video: # 大于指定帧数则使用均匀采样
                selected_idxs=np.linspace(0,this_feature_nums,num=self.n_flames_per_video)
                this_features=this_features[selected_idxs,:]
            return_features.append(this_features)
            captions=[0 for _ in range(self.n_words_per_caption+2)]
            i=0
            captions[0]=self.vacabs.word2idx['<bos>']         # <bos> word1 word2 word3 .... word20 <eos> 
            for w in self.captions[idx].lower().split(' '):
                if w not in self.vacabs.word2idx:
                    captions[i+1]=self.vacabs.word2idx['<unk>']
                else:
                    captions[i+1]=self.vacabs.word2idx[w]
                i+=1
                if i>self.n_words_per_caption:
                    break
            captions[i]=self.vacabs.word2idx['<eos>']

            return_captions.append(captions)

        return np.array(return_features,dtype=np.float32),np.array(return_captions,dtype=np.int32)

    def get_test_batch(self,batch_size):
        '''评测的时候使用'''
        return_features = []
        return_captions = []
        videos=[]
        videos_path_unique=list(set(self.video_paths)) #去除重复的视频
        count=0
        for idx in range(0,len(videos_path_unique)):
            this_path = os.path.basename(videos_path_unique[idx]).split('.')[0]
            videos.append(this_path)
            feature_path = os.path.join(self.data_dir, this_path + '.npy')
            if os.path.exists(feature_path):
                this_features = np.load(feature_path, allow_pickle=True)
            else:
                raise ValueError('feature path not existed in class %s' % self.__class__.__name__)
            this_features = np.vstack(this_features)
            this_feature_nums, dims_feature = this_features.shape
            if this_feature_nums < self.n_flames_per_video:  # 小于需要的帧数则填充0
                this_features = np.vstack([this_features,
                                           np.zeros(shape=(self.n_flames_per_video - this_feature_nums, dims_feature))])
            if this_feature_nums > self.n_flames_per_video:  # 大于指定帧数则使用均匀采样
                selected_idxs = np.linspace(0, this_feature_nums, num=self.n_flames_per_video)
                this_features = this_features[selected_idxs, :]
            return_features.append(this_features)
            captions=self.dataset[self.dataset['Video_path']==videos_path_unique[idx]]['Description'].values
            captions=list(replace_map(captions))
            captions=[caption.lower() for caption in captions]
            return_captions.append(captions)
            count+=1
            if count==batch_size:
                yield np.array(return_features, dtype=np.float32), return_captions,videos
                return_features,return_captions,videos=[],[],[]
                count=0

        if count!=0:
            yield np.array(return_features, dtype=np.float32), return_captions,videos

if __name__ == '__main__':
    SAVEPATH = '../features/Features'
    dataloader=DataLoader(CSV_PATH,data_dir=SAVEPATH)
    x,y=dataloader.get_batch(batch_size=2)
    print(x.shape)
    print(y)
    for x_test,y_test,videos in dataloader.get_test_batch(batch_size=2):
        print(x_test.shape)
        print(videos)
        for caption in y_test[0]:
            print(caption)
        break




