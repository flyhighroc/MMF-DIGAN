# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:11:03 2020
DI-GAN model
@author: Pengfei Fan
"""

import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Concatenate, Dropout
from tensorflow.keras.layers import LeakyReLU, Embedding
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import time
import datetime

from layers import *
from utils import *

datapath = '../data/'

savepath='./save/'  #sl + cvaegan, dmd as noise, no new sample z_p, x1_f_, x1_p_ and x1_r in gan
if os.path.exists(savepath) == False:
    os.makedirs(savepath)
    
image_path = savepath+'/images/'
if os.path.exists(image_path) == False:
    os.makedirs(image_path)
    
model_path = savepath+'/models/'
if os.path.exists(model_path) == False:
    os.makedirs(model_path)
    
evaluationl_path = savepath+'evaluation/'
if os.path.exists(evaluationl_path) == False:
    os.makedirs(evaluationl_path)

class DINet():
    def __init__(self):
        self.img_rows = 96
        self.img_cols = 96
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dmd_dim = 24 * 24
        self.num_classes = 10
        self.classes_dim = 100
        
        self.sLosses = []
        self.eLosses = []
        self.gLosses = []
        self.dLosses = []
        self.cLosses = []
        
        self.eAccuracies = []
        self.gAccuracies = []
        self.dAccuracies = []
        self.cAccuracies = []
        
        self.e_time = []
        
        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        self.decoder = None
        self.encoder = None
        self.generator = None
        self.discriminator = None
        self.classifier = None
        self.dec_trainer = None
        self.enc_trainer = None
        self.gen_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.build_model()
        
    def build_model(self):
        self.decoder = self.build_decoder()
        self.encoder = self.build_encoder()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.classifier = self.build_classifier()

        # Algorithm
        
        x1_r = Input(shape=self.img_shape, name='Transmitted_speckle')
        x2_r = Input(shape=self.img_shape, name='Reflected_speckle')
        c_r = Input(shape=(1,), name='Bending_state')
        z_r = Input(shape=(self.dmd_dim,), name='DMD_pattern')
        
        x2_ = self.decoder([c_r, z_r]) 

        s_loss = DecoderLossLayer()([x2_r, x2_])
        
        z_ = self.encoder([c_r, x1_r]) 

        e_loss = EncoderLossLayer()([z_r, z_])
        
        x1_f_ = self.generator([z_, x2_]) 
        x1_p_ = self.generator([z_r, x2_r])
        
        g_loss = GeneratorLossLayer()([x1_r, x1_f_, x1_p_])
        
        y_r_ = self.discriminator(x1_r)
        y_f_ = self.discriminator(x1_f_)
        y_p_ = self.discriminator(x1_p_)
        
        d_loss = DiscriminatorLossLayer()([y_r_, y_f_, y_p_])
    
        c_r_ = self.classifier(x2_r)

        c_loss = ClassifierLossLayer()([c_r, c_r_])
        
        # Build decoder trainer
        set_trainable(self.decoder, True)
        set_trainable(self.encoder, False)
        set_trainable(self.generator, False)
        set_trainable(self.discriminator, False)
        set_trainable(self.classifier, False)

        self.dec_trainer = Model(inputs=[x1_r, x2_r, c_r, z_r],
                                 outputs=[g_loss, s_loss])
        self.dec_trainer.compile(loss=[zero_loss, zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.dec_trainer.summary()        

        # Build encoder trainer
        set_trainable(self.decoder, False)
        set_trainable(self.encoder, True)
        set_trainable(self.generator, False)
        set_trainable(self.discriminator, False)
        set_trainable(self.classifier, False)

        self.enc_trainer = Model(inputs=[x1_r, x2_r, c_r, z_r],
                                outputs=[g_loss, e_loss])
        self.enc_trainer.compile(loss=[zero_loss, zero_loss],
                                optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                metrics=[encoder_accuracy(z_r, z_)])
        self.enc_trainer.summary()

        # Build generator trainer
        set_trainable(self.decoder, False)
        set_trainable(self.encoder, False)
        set_trainable(self.generator, True)
        set_trainable(self.discriminator, False)
        set_trainable(self.classifier, False)

        self.gen_trainer = Model(inputs=[x1_r, x2_r, c_r, z_r],
                                 outputs=[g_loss])
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy2(y_p_, y_f_)])
#                                 metrics=[generator_accuracy1(y_f_)])   #only x1_f_ and x1_r in gan, no x1_p_
        self.gen_trainer.summary()

        # Build discriminator trainer
        set_trainable(self.decoder, False)
        set_trainable(self.encoder, False)
        set_trainable(self.generator, False)
        set_trainable(self.discriminator, True)
        set_trainable(self.classifier, False)

        self.dis_trainer = Model(inputs=[x1_r, x2_r, c_r, z_r],
                                 outputs=[d_loss])
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[discriminator_accuracy(y_r_, y_f_, y_p_)])
        self.dis_trainer.summary()

        # Build classifier trainer
        set_trainable(self.decoder, False)
        set_trainable(self.encoder, False)
        set_trainable(self.generator, False)
        set_trainable(self.discriminator, False)
        set_trainable(self.classifier, True)

        self.cls_trainer = Model(inputs=[x2_r, c_r],
                                 outputs=[c_loss])
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[classifier_accuracy(c_r, c_r_)])
        self.cls_trainer.summary()

    def build_decoder(self):
        
        label = Input(shape=(1,), name='Bending_state')
        label_embedding = Flatten()(Embedding(self.num_classes, self.classes_dim)(label))
        
        dmd = Input(shape=(self.dmd_dim,), name='DMD_pattern')
        
        model_input = Concatenate()([label_embedding, dmd])
        
        x = dense_layer(model_input, 1024)
        x = dense_layer(x, 2048)        
        x = dense_layer(x, 4096)   
        x = Dense(np.prod(self.img_shape), activation='tanh')(x)
        img_ = Reshape(self.img_shape)(x)
        
        model = Model([label, dmd], img_, name='Decoder')
#        model.summary()
#        plot_model(model, to_file=savepath+'decoder_model.png', show_shapes=True)

        return model

    def build_encoder(self):
            
        label = Input(shape=(1,), name='Bending_state')
        img = Input(shape=self.img_shape, name='Transmitted_speckle')    
        
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        
    #    c = dense_layer(label, np.prod(img_shape))
        c = Reshape(self.img_shape)(label_embedding)
        
        x = Concatenate(axis=-1)([img, c])
        
        x = conv2d_layer(x, 128)
        x = conv2d_layer(x, 256)
        x = conv2d_layer(x, 256)
        x = conv2d_layer(x, 512)
        x = Flatten()(x)
               
        x = dense_layer(x, 128*128)       
                   
        dmd_ = Dense(self.dmd_dim, activation="sigmoid")(x)
        
        model = Model([label, img], dmd_, name='Encoder')
#        model.summary()
#        plot_model(model, to_file=savepath+'encoder_model.png', show_shapes=True)
    
        return model
                
    def build_generator(self):

        d = Input(shape=(self.dmd_dim,), name='DMD_pattern')
        dmd = dense_layer(d, 1024)
        dmd = dense_layer(dmd, 2048)        
        dmd = dense_layer(dmd, 4096)   
        dmd = Dense(np.prod(self.img_shape))(dmd)
        dmd = LeakyReLU(alpha=0.2)(dmd)
        dmd = Reshape(self.img_shape)(dmd)
        
        x = Input(shape=self.img_shape, name='Reflected_speckle')
        
        # Merged image input       
        d0 = Concatenate(axis=-1)([dmd, x])

        # Downsampling
        d1 = conv2d_layer(d0, self.gf)
        d2 = conv2d_layer(d1, self.gf*2)
        d3 = conv2d_layer(d2, self.gf*4)
        d4 = conv2d_layer(d3, self.gf*8)
        d5 = conv2d_layer(d4, self.gf*16)

        # Upsampling
        u1 = deconv2d_layer(d5, d4, self.gf*8)
        u2 = deconv2d_layer(u1, d3, self.gf*4)
        u3 = deconv2d_layer(u2, d2, self.gf*2)
        u4 = deconv2d_layer(u3, d1, self.gf)

        u5 = UpSampling2D(size=2)(u4)
        output_img = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(u5)

        model = Model([d, x], output_img, name='Generator')
#        model.summary()   
#        plot_model(model, to_file=savepath+'generator_model.png', show_shapes=True)
        
        return model
        
    def build_discriminator(self):
                
        img = Input(shape=self.img_shape, name='Transmitted_speckle')
        
        y = conv2d_layer(img, 16)
        y = conv2d_layer(y, 32)
        y = conv2d_layer(y, 64)
        y = conv2d_layer(y, 128)
        y = Flatten()(y)
        
        # Extract feature representation
        features = dense_layer(y, 1024)

        # Determine validity of the image
        validity = Dense(1, activation="sigmoid")(features)
 
        model = Model(img, validity, name='Discriminator')
#        model.summary()
#        plot_model(model, to_file=savepath+'discriminator_model.png', show_shapes=True)

        return model
    
    def build_classifier(self):
                
        img = Input(shape=self.img_shape, name='Reflected_speckle')
        
        y = conv2d_layer(img, 16)
        y = conv2d_layer(y, 32)
        y = conv2d_layer(y, 64)
        y = conv2d_layer(y, 128)
        y = Flatten()(y)
        
        # Extract feature representation
        features = dense_layer(y, 1024)

        # Determine label of the image
        label = Dense(self.num_classes, activation="softmax")(features)

        model = Model(img, label, name='Classifier')
#        model.summary()
#        plot_model(model, to_file=savepath+'classifier_model.png', show_shapes=True)

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):
        
        # Load the dataset
        x1 = np.load(datapath+'x_T_all.p.npy')
        print('x1 shape:', x1.shape)
        x2 = np.load(datapath+'x_R_all.p.npy')
        print('x2 shape:', x2.shape)
        y = np.load(datapath+'inputs.p.npy')
        y = y.reshape(y.shape[0],y.shape[1]*y.shape[2])
        y_all = np.tile(y, (10,1))
        print('dmd shape:', y_all.shape)
        label_all = np.array(list(range(10))).repeat(12000)
        label_all = label_all.reshape(-1, 1)
        print('label shape:', label_all.shape)
        
        # Configure inputs
        x1 = (x1.astype(np.float32) - 127.5) / 127.5
        x2 = (x2.astype(np.float32) - 127.5) / 127.5
        
        X1_train, X1_test, X2_train, X2_test, y_train, y_test, label_train, label_test = train_test_split(x1, x2, y_all, label_all, test_size=0.1, random_state=0)
        
        start_time_total = datetime.datetime.now()
        
        for epoch in range(1, epochs+1):
            
            start_time = time.time()
         
            idx_r = np.random.randint(0, X1_train.shape[0], batch_size)
            # Select a random batch of images
            x1_r = X1_train[idx_r]
            x2_r = X2_train[idx_r]
            # Image labels. 0-9 
            c_r = label_train[idx_r]
            # Sample dmd patterns 
            z_r = y_train[idx_r]
            
            x1_dummy = np.zeros(x1_r.shape, dtype='float32')
            x2_dummy = np.zeros(x2_r.shape, dtype='float32')
            c_dummy = np.zeros(c_r.shape, dtype='float32')
            z_dummy = np.zeros(z_r.shape, dtype='float32')
            y_dummy = np.zeros((batch_size, 1), dtype='float32')
            
            # Train decoder
            _, _, s_loss = self.dec_trainer.train_on_batch([x1_r, x2_r, c_r, z_r], [x1_dummy, x2_dummy])
            
            # Train encoder
            _, _, e_loss, _, e_acc = self.enc_trainer.train_on_batch([x1_r, x2_r, c_r, z_r], [x1_dummy, z_dummy])
    
            # Train generator
            g_loss, g_acc = self.gen_trainer.train_on_batch([x1_r, x2_r, c_r, z_r], x1_dummy)
                
            # Train discriminator
            d_loss, d_acc = self.dis_trainer.train_on_batch([x1_r, x2_r, c_r, z_r], y_dummy)

            # Train classifier
            c_loss, c_acc = self.cls_trainer.train_on_batch([x2_r, c_r], c_dummy)           
            
            elapsed_time = time.time() - start_time

            # Plot the progress
            print ("%d [S loss: %f] [E loss: %f, acc: %.2f%%] [G loss: %f, acc: %.2f%%] [D loss: %f, acc: %.2f%%] [C loss: %f, acc: %.2f%%] " % (epoch, s_loss, e_loss, 100*e_acc, g_loss, 100*g_acc, d_loss, 100*d_acc, c_loss, 100*c_acc))

            # Store loss of most recent batch from this epoch
            self.sLosses.append(s_loss)
            self.eLosses.append(e_loss)
            self.gLosses.append(g_loss)            
            self.dLosses.append(d_loss)
            self.cLosses.append(c_loss)

            self.eAccuracies.append(e_acc)
            self.gAccuracies.append(g_acc)            
            self.dAccuracies.append(d_acc)
            self.cAccuracies.append(c_acc)         

            self.e_time.append(elapsed_time)
                                                        
            # If at save interval => save generated image samples
            if epoch == 1 or epoch % sample_interval == 0:
#                self.sample_images(epoch)
                self.save_model(epoch)

        elapsed_time_total = datetime.datetime.now() - start_time_total
        print ('Total training time:', elapsed_time_total)
        
        np.save(evaluationl_path+'sLosses.p', self.sLosses)
        np.save(evaluationl_path+'eLosses.p', self.eLosses)        
        np.save(evaluationl_path+'gLosses.p', self.gLosses)        
        np.save(evaluationl_path+'dLosses.p', self.dLosses)
        np.save(evaluationl_path+'cLosses.p', self.cLosses)
        
        np.save(evaluationl_path+'eAccuracies.p', self.eAccuracies)        
        np.save(evaluationl_path+'gAccuracies.p', self.gAccuracies)        
        np.save(evaluationl_path+'dAccuracies.p', self.dAccuracies)
        np.save(evaluationl_path+'cAccuracies.p', self.cAccuracies)
        
        np.save(evaluationl_path+'elapsedTime.p', self.e_time)
        
#    def sample_images(self, epoch):
#        r, c = 10, 10
#        sampled_dmds = np.random.randint(0,2,(r * c, self.dmd_dim))
#        sampled_imgs_B = np.array([num for _ in range(r) for num in range(c)])
#        gen_imgs = self.generator.predict([sampled_dmds, sampled_imgs_B])
#        # Rescale images 0 - 1
#        gen_imgs = 0.5 * gen_imgs + 0.5
#
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig(image_path+"%d.png" % epoch)
#        plt.close()
        
    # Save the networks (and weights) for later use
    def save_model(self, epoch):
        self.decoder.save(model_path+'DINet_decoder_epoch_%d.h5' % epoch)
        self.encoder.save(model_path+'DINet_encoder_epoch_%d.h5' % epoch)     
        self.generator.save(model_path+'DINet_generator_epoch_%d.h5' % epoch)       
        self.discriminator.save(model_path+'DINet_discriminator_epoch_%d.h5' % epoch)
        self.classifier.save(model_path+'DINet_classifier_epoch_%d.h5' % epoch)
                
                
if __name__ == '__main__':
    digan = DINet()
    digan.train(epochs=200000, batch_size=32, sample_interval=20000)