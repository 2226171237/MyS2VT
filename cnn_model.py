# -*- coding=utf-8 -*-
import tensorflow as tf
import os

class CNN:
    model_paths={
        'vgg16':os.path.join(os.getcwd(),'./cnn_weights/vgg16_notop.h5'),
        'resnet50':os.path.join(os.getcwd(),'./cnn_weights/resnet50_notop.h5'),
        'mobilenetv2':os.path.join(os.getcwd(), './cnn_weights/mobilenet_v2_notop.h5')
    }
    def __init__(self,net='mobilenetv2'):
        assert net in ['vgg16','resnet50','mobilenetv2']
        self.net_dicts={
            'vgg16':tf.keras.applications.vgg16.VGG16,
            'resnet50':tf.keras.applications.resnet50.ResNet50,
            'mobilenetv2':tf.keras.applications.mobilenet_v2.MobileNetV2
        }
        # include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000
        self.model=self.net_dicts[net](input_shape=(224,224,3),include_top=False,weights=None,pooling='avg')
        self.model.trainable=False
        self.weight_path=self.model_paths[net]
        # for layer in self.model.layers:
        #     print("name:",layer.name,', output shape:',layer.output_shape)
        self.model.summary()
        self.have_load_weights=False

    def load_weights(self,weight_path=None):
        weight_path=weight_path if weight_path else self.weight_path
        assert os.path.exists(weight_path)
        self.model.load_weights(weight_path)
        self.have_load_weights=True

    def get_features(self,flames):
        if not self.have_load_weights:
            raise ValueError('your model have not load weights.')
        else:
            return self.model.predict_on_batch(flames)



if __name__=='__main__':
    import numpy as np
    import time
    x=np.random.randn(64,224,224,3)
    cnn=CNN()
    cnn.load_weights()
    start=time.time()
    fs=cnn.get_features(x)
    print(fs.shape,fs.dtype,'time:',time.time()-start)
