# -*- coding=utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class CaptionGenerator(tf.keras.Model):
    def __init__(self,n_words,batch_size,dim_feature=512,dim_hidden=512,n_video_lstm=80,
                 n_caption_lstm=20,bias_init_vector=None):
        super(CaptionGenerator, self).__init__()
        self.n_words=n_words
        self.dim_feature=dim_feature
        self.dim_hidden=dim_hidden
        self.n_video_lstm=n_video_lstm
        self.n_caption_lstm=n_caption_lstm
        self.batch_size=batch_size
        self.wordEmbed = tf.Variable(tf.random_uniform_initializer(-0.1, 0.1)(shape=(n_words, dim_hidden)),name='wordEmbed')

        self.dense_feature=keras.layers.Dense(units=dim_hidden,name='dense_feature')
        self.lstm1=keras.layers.LSTMCell(units=dim_hidden,name='lstm_video')
        self.lstm2=keras.layers.LSTMCell(units=dim_hidden,name='lstm_caption')

        self.dense_output=keras.layers.Dense(units=n_words,
                                             bias_initializer=keras.initializers.Constant(bias_init_vector if bias_init_vector else 0.),
                                             name='dense_output')

    def call(self,X,Y=None,Y_mask=None):
        if Y is not None:
            return self.train(X,Y,Y_mask)  # loss
        else:
            return self.predict(X)   # result

    def train(self,X,Y,Y_mask):
        self.state1 = self.lstm1.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        self.state2 = self.lstm2.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        self.padding = tf.zeros([self.batch_size, self.dim_hidden])
        X = tf.reshape(X, shape=(-1, self.dim_feature))  # (batch_size*T,dim_feature)
        X = self.dense_feature(X)  # (batch_size*T,dim_hidden)
        X = tf.reshape(X, shape=(self.batch_size, -1, self.dim_hidden))
        # encoding video
        losses=0.0
        for i in range(self.n_video_lstm):
            output1, self.state1 = self.lstm1(X[:, i, :], self.state1)
            output2, self.state2 = self.lstm2(tf.concat([output1, self.padding], 1), self.state2)

        # decoding
        for i in range(self.n_caption_lstm + 1):
            with tf.device('cpu:0'):
                current_embed = tf.nn.embedding_lookup(self.wordEmbed, Y[:, i])
            output1, self.state1 = self.lstm1(self.padding, self.state1)
            output2, self.state2 = self.lstm2(tf.concat([output1, current_embed], 1), self.state2)

            labels=Y[:,i+1] #当前的label是下一个word
            onehot_labels=tf.one_hot(labels,depth=self.n_words)

            logit_words=self.dense_output(output2)
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,logits=logit_words)
            cross_entropy=cross_entropy*Y_mask[:,i]
            current_loss=tf.reduce_mean(cross_entropy)
            losses+=current_loss
        return losses

    def predict(self,X):
        self.state1 = self.lstm1.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        self.state2 = self.lstm2.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        self.padding = tf.zeros([self.batch_size, self.dim_hidden])
        X = tf.reshape(X, shape=(-1, self.dim_feature))  # (batch_size*T,dim_feature)
        X = self.dense_feature(X)  # (batch_size*T,dim_hidden)
        X = tf.reshape(X, shape=(self.batch_size, -1, self.dim_hidden))
        # encoding video
        for i in range(self.n_video_lstm):
            output1, self.state1 = self.lstm1(X[:, i, :], self.state1)
            output2, self.state2 = self.lstm2(tf.concat([output1, self.padding], 1), self.state2)

        # decoding
        generated_words=[]
        for i in range(self.n_caption_lstm + 1):
            if i==0:
                with tf.device('cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.wordEmbed, tf.ones([self.batch_size],dtype=tf.int64))
            output1, self.state1 = self.lstm1(self.padding, self.state1)
            output2, self.state2 = self.lstm2(tf.concat([output1, current_embed], 1), self.state2)


            logit_words = self.dense_output(output2)
            max_prob_index=tf.argmax(logit_words,axis=-1)
            with tf.device('cpu:0'):
                current_embed=tf.nn.embedding_lookup(self.wordEmbed,max_prob_index)
            generated_words.append(max_prob_index.numpy())

        return np.array(generated_words)
    

if __name__ == '__main__':
    X=np.random.rand(3,80,512)
    X_mask=np.ones_like(X)
    Y=np.random.randint(0,100,size=(3,22))
    Y_mask=np.ones_like(Y)
    rnn=CaptionGenerator(100,3)

    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    for i in range(20):
        with tf.GradientTape() as tape:
            loss=rnn(X,Y,Y_mask)
        grads=tape.gradient(loss,rnn.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,rnn.variables))
        if (i+1)%10==0:
            print('%d/100: loss %f' % (i+1,loss.numpy()))
    rnn.summary()
    result=rnn(X)
    print(result.T)
