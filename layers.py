import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dense, Concatenate
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras import backend as K

class DecoderLossLayer(Layer):
    __name__ = 'decoder_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DecoderLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred):
        return 10**3 * K.mean(tf.keras.losses.mean_squared_error(x_true, x_pred))

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        loss = self.lossfun(x_true, x_pred)
        self.add_loss(loss, inputs=inputs)

        return x_true
    
class EncoderLossLayer(Layer):
    __name__ = 'encoder_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(EncoderLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_true, z_pred):
        return 10**3 * K.mean(tf.keras.losses.binary_crossentropy(z_true, z_pred))

    def call(self, inputs):
        z_true = inputs[0]
        z_pred = inputs[1]
        loss = self.lossfun(z_true, z_pred)
        self.add_loss(loss, inputs=inputs)

        return z_true

class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_pred):
        return K.mean(tf.keras.losses.sparse_categorical_crossentropy(c_true, c_pred))

    def call(self, inputs):
        c_true = inputs[0]
        c_pred = inputs[1]
        loss = self.lossfun(c_true, c_pred)
        self.add_loss(loss, inputs=inputs)

        return c_true

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_f, y_fake_p):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = tf.keras.losses.binary_crossentropy(y_pos, y_real)
        loss_fake_f = tf.keras.losses.binary_crossentropy(y_neg, y_fake_f)
        loss_fake_p = tf.keras.losses.binary_crossentropy(y_neg, y_fake_p)
        return K.mean(loss_real + loss_fake_f + loss_fake_p)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        y_fake_p = inputs[2]
        loss = self.lossfun(y_real, y_fake_f, y_fake_p)
        self.add_loss(loss, inputs=inputs)

        return y_real
   
class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred_f, x_pred_p):
        loss_f = tf.keras.losses.mean_squared_error(x_true, x_pred_f)
        loss_p = tf.keras.losses.mean_squared_error(x_true, x_pred_p)
        return 10**3 * K.mean(loss_f + loss_p)
 
    def call(self, inputs):
        x_true = inputs[0]
        x_pred_f = inputs[1]
        x_pred_p = inputs[2]
        loss = self.lossfun(x_true, x_pred_f, x_pred_p)
        self.add_loss(loss, inputs=inputs)

        return x_true

def dense_layer(layer_input, units, bn=True, dropout_rate=0.4):
    """Dense layer"""
    d = Dense(units)(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    if dropout_rate:
        d = Dropout(dropout_rate)(d)
    return d    

def conv2d_layer(layer_input, filters, f_size=3, bn=True, dropout_rate=0.4):
    """Conv2d layers"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    if dropout_rate:
        d = Dropout(dropout_rate)(d)
    return d

def deconv2d_layer(layer_input, skip_input, filters, f_size=3, bn=True, dropout_rate=0.4):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
    u = LeakyReLU(alpha=0.2)(u)
    if bn:
        u = BatchNormalization(momentum=0.8)(u)            
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = Concatenate()([u, skip_input])
    return u