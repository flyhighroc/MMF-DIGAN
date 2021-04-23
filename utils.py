import tensorflow.keras
from tensorflow.keras import backend as K

def set_trainable(model, train):
    """
    Enable or disable training for the model
    args:
        model(?):
        train(?):
    """
    model.trainable = train
    for l in model.layers:
        l.trainable = train


def zero_loss(y_true, y_pred):
    """
    args:
        y_true():
        y_pred():
    """
    return K.zeros_like(y_true)

def sample_normal(args):
    """

    """
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return "%d sec" %s
    else:
        return "%d min %d sex" %(m, s)

def encoder_accuracy(z_t, z_p):
    def accfun(y0, y1):
        loss = K.mean(tf.keras.metrics.binary_accuracy(z_t, z_p))
        return loss

    return accfun

def discriminator_accuracy(x_r, x_f, x_p):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_r)
        x_neg = K.zeros_like(x_r)
        loss_r = K.mean(tf.keras.metrics.binary_accuracy(x_pos, x_r))
        loss_f = K.mean(tf.keras.metrics.binary_accuracy(x_neg, x_f))
        loss_p = K.mean(tf.keras.metrics.binary_accuracy(x_neg, x_p))
        return (1.0 / 3.0) * (loss_r + loss_p + loss_f)

    return accfun

def generator_accuracy1(x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_f)
        loss_f = K.mean(tf.keras.metrics.binary_accuracy(x_pos, x_f))
        return loss_f

    return accfun

def generator_accuracy2(x_p, x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_p)
        loss_p = K.mean(tf.keras.metrics.binary_accuracy(x_pos, x_p))
        loss_f = K.mean(tf.keras.metrics.binary_accuracy(x_pos, x_f))
        return 0.5 * (loss_p + loss_f)

    return accfun

def classifier_accuracy(c_t, c_p):
    def accfun(y0, y1):
        loss = K.mean(tf.keras.metrics.sparse_categorical_accuracy(c_t, c_p))
        return loss

    return accfun