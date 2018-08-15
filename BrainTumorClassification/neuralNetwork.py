import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os, tflearn
import numpy as np

#Now we have batch created let us load tensorflow 
def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]], name="x"  )


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.int32, shape=(None,),name="y")


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name="keep_prob")


def neural_is_training():
    return tf.placeholder(tf.bool, name="is_training")

def flatten(x_tensor):
    dim_list = x_tensor.get_shape().as_list()
   
    return tf.reshape(x_tensor, [-1, dim_list[1] * dim_list[2] * dim_list[3] ])

def fully_conn(x_tensor, num_outputs):

    num_in = x_tensor.get_shape().as_list()[1]
    
    weights = tf.Variable(tf.random_normal([num_in, num_outputs], stddev=0.05))
    
    bias = tf.Variable(tf.zeros(num_outputs))
    
    fully_conn = tf.add(tf.matmul(x_tensor, weights), bias)
    
    fully_conn = tf.nn.relu(fully_conn)
    
    return fully_conn

def output(x_tensor, num_outputs):
    num_in = x_tensor.get_shape().as_list()[1]
    weights = tf.Variable(tf.random_normal([num_in, num_outputs],stddev=0.05))
    bias = tf.Variable(tf.zeros(num_outputs))
    
    fully_conn = tf.add(tf.matmul(x_tensor, weights), bias)
    
    return fully_conn


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):

    channels = x_tensor.get_shape().as_list()[3]

    weights = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], channels, conv_num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    
    conv_layer = tf.nn.conv2d(x_tensor, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    
    conv_layer = tf.nn.relu(conv_layer)
    
    
    #for i in range(2,3,1):
    channels = conv_layer.get_shape().as_list()[3]

    #conv_num_outputs_1 = conv_num_outputs * i

    weights1 = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], channels, conv_num_outputs*2], stddev=0.05))
    bias1 = tf.Variable(tf.zeros(conv_num_outputs*2))

    conv_layer = tf.nn.conv2d(conv_layer, weights1, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')

    conv_layer = tf.nn.bias_add(conv_layer, bias1)

    conv_layer = tf.nn.relu(conv_layer)
#    if conv_num_outputs > 4:
    conv_layer = tf.nn.max_pool(conv_layer, [1, pool_ksize[0], pool_ksize[1], 1],\
                                [1, pool_strides[0], pool_strides[1], 1], padding='SAME'  )
    # TODO: Implement Function
    return conv_layer 


def conv_net(x, keep_prob):
    conv_ksize = (5, 5)
    conv_strides = (1, 1)
    pool_ksize = (2, 2)
    pool_strides = (2, 2)
    conv_num_outputs = 4
    
    print("The ConvNet Model")
    
    # CNN layer 1 -- 512x512x1 to 256x256x4
    conv_layer = conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    #conv_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    # CNN layer 2 -- 256x256x4 to 128x128x16
    conv_layer = conv2d_maxpool(conv_layer, 16, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    #conv_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    # CNN layer 3 -- 128x128x16 to 64x64x64
    #conv_layer = conv2d_maxpool(conv_layer, 64, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_layer = conv2d_maxpool(conv_layer, 48, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    #conv_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    # CNN layer 4 - 64x64x64 to 32x32x256
    #conv_layer = conv2d_maxpool(conv_layer, 128, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    # 2x2x512 to 2048
    #print("conv1")
    conv_layer = flatten(conv_layer)

    # Dropout -- 3072 to (keep_prob * 3072)
    fully_conn_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    # Fully connected layer 1 -- (keep_prob * 3072) to 512
    #fully_conn_layer = fully_conn(conv_layer, 512)
    
    # Fully connected layer 2 -- 512 to 128
    #print("full_conv1")

    #fully_conn_layer = fully_conn(fully_conn_layer, 4096)
    #4096 memory overrun
    
    fully_conn_layer = fully_conn(fully_conn_layer, 1536)
    fully_conn_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    fully_conn_layer = fully_conn(fully_conn_layer, 384)
    fully_conn_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    fully_conn_layer = fully_conn(fully_conn_layer, 96)
    fully_conn_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    #fully_conn_layer = tf.nn.dropout(fully_conn_layer, keep_prob)
    
    fully_conn_layer = fully_conn(fully_conn_layer, 24)
    fully_conn_layer = tf.nn.dropout(conv_layer, keep_prob)
    
    # Output layer -- 128 to 10
    out = output(fully_conn_layer, 3)
    
    return out

########################################################################################################################################################################################################################################################################################

def googlenet(inputs,
              dropout_keep_prob=0.4,
              num_classes=1000,
              is_training=True,
              restore_logits = None,
              scope=''):
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.merge_ops import merge
    from tflearn.layers.estimator import regression
    
    print("The GoogleNet Model")
    
    conv1_7_7 = conv_2d(inputs, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


    inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


    inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

    inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
    inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
    inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
    inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
    inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

    inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
    inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
    inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
    inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


    inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
    inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


    inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
    inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

    pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    # output stage
    logits = fully_connected(pool5_7_7, 3 ,activation=None)

    return logits


'''
print("altered GoogleNet")
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops

def get_inception_layer( inputs, conv11_size, conv33_11_size, conv33_size,
                         conv55_11_size, conv55_size, pool11_size ):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d( inputs, conv11_size, [ 1, 1 ] )
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d( inputs, conv33_11_size, [ 1, 1 ] )
        conv33 = layers.conv2d( conv33_11, conv33_size, [ 3, 3 ] )
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d( inputs, conv55_11_size, [ 1, 1 ] )
        conv55 = layers.conv2d( conv55_11, conv55_size, [ 5, 5 ] )
    with tf.variable_scope("pool_proj"):
        pool_proj = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
        pool11 = layers.conv2d( pool_proj, pool11_size, [ 1, 1 ] )
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool11])
    return tf.concat([conv11, conv33, conv55, pool11], 3)

def aux_logit_layer( inputs, num_classes, is_training ):
    with tf.variable_scope("pool2d"):
        pooled = layers.avg_pool2d(inputs, [ 5, 5 ], stride = 3 )
    with tf.variable_scope("conv11"):
        conv11 = layers.conv2d( pooled, 128, [1, 1] )
    with tf.variable_scope("flatten"):
        conv11Shape = conv11.get_shape().as_list()
        #end_points['reshape'] = tf.reshape( end_points['pool4'], [-1, pool4Shape[1] * pool4Shape[2] * pool4Shape[3]] )        
        #flat = tf.reshape( conv11, [-1, 2048] )
        flat = tf.reshape( conv11, [-1, conv11Shape[1] * conv11Shape[2] * conv11Shape[3]] )
    with tf.variable_scope("fc"):
        fc = layers.fully_connected( flat, 1024, activation_fn=None )
    with tf.variable_scope("drop"):
        drop = layers.dropout( fc, 0.3, is_training = is_training )
    with tf.variable_scope( "linear" ):
        linear = layers.fully_connected( drop, num_classes, activation_fn=None )
    with tf.variable_scope("soft"):
        soft = tf.nn.softmax( linear )
    return soft

def googlenet(inputs,
              dropout_keep_prob=0.4,
              num_classes=3,
              is_training=True,
              restore_logits = None,
              scope=''):
    
    #Implementation of https://arxiv.org/pdf/1409.4842.pdf
    

    end_points = {}
        
    with tf.name_scope( scope, "googlenet", [inputs] ):
        with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
            end_points['conv0'] = layers.conv2d( inputs, 64, [ 8, 8 ], stride = 2, scope = 'conv0' )
            end_points['pool0'] = layers.max_pool2d(end_points['conv0'], [3, 3], scope='pool0')
            end_points['conv1_a'] = layers.conv2d( end_points['pool0'], 64, [ 1, 1 ], scope = 'conv1_a' )
            end_points['conv1_b'] = layers.conv2d( end_points['conv1_a'], 192, [ 3, 3 ], scope = 'conv1_b' )
            end_points['pool1'] = layers.max_pool2d(end_points['conv1_b'], [ 3, 3 ], scope='pool1')

            #end_points['conv2_a'] = layers.conv2d( end_points['pool01'], 192, [ 1, 1 ], scope = 'conv2_a' )
            #end_points['conv2_b'] = layers.conv2d( end_points['conv2_a'], 256, [ 3, 3 ], scope = 'conv2_b' )
            #end_points['pool1'] = layers.max_pool2d(end_points['conv2_b'], [ 3, 3 ], scope='pool1')
            
            with tf.variable_scope("inception_3a"):
                end_points['inception_3a'] = get_inception_layer( end_points['pool1'], 64, 96, 128, 16, 32, 32 )

            with tf.variable_scope("inception_3b"):
                end_points['inception_3b'] = get_inception_layer( end_points['inception_3a'], 128, 128, 192, 32, 96, 64 )

            end_points['pool2'] = layers.max_pool2d(end_points['inception_3b'], [ 3, 3 ], scope='pool2')

            with tf.variable_scope("inception_4a"):
                end_points['inception_4a'] = get_inception_layer( end_points['pool2'], 192, 96, 208, 16, 48, 64 )

            with tf.variable_scope("aux_logits_1"):
                end_points['aux_logits_1'] = aux_logit_layer( end_points['inception_4a'], num_classes, is_training )

            with tf.variable_scope("inception_4b"):
                end_points['inception_4b'] = get_inception_layer( end_points['inception_4a'], 160, 112, 224, 24, 64, 64 )

            with tf.variable_scope("inception_4c"):
                end_points['inception_4c'] = get_inception_layer( end_points['inception_4b'], 128, 128, 256, 24, 64, 64 )

            with tf.variable_scope("inception_4d"):
                end_points['inception_4d'] = get_inception_layer( end_points['inception_4c'], 112, 144, 288, 32, 64, 64 )

            with tf.variable_scope("aux_logits_2"):
                end_points['aux_logits_2'] = aux_logit_layer( end_points['inception_4d'], num_classes, is_training )

            with tf.variable_scope("inception_4e"):
                end_points['inception_4e'] = get_inception_layer( end_points['inception_4d'], 256, 160, 320, 32, 128, 128 )

            end_points['pool3'] = layers.max_pool2d(end_points['inception_4e'], [ 3, 3 ], scope='pool3')

            #expanding the belly
            with tf.variable_scope("inception_5a"):
                end_points['inception_5a'] = get_inception_layer( end_points['pool3'], 192, 96, 208, 16, 48, 64 )

            with tf.variable_scope("aux_logits_3"):
                end_points['aux_logits_3'] = aux_logit_layer( end_points['inception_5a'], num_classes, is_training )

            with tf.variable_scope("inception_5b"):
                end_points['inception_5b'] = get_inception_layer( end_points['inception_5a'], 160, 112, 224, 24, 64, 64 )

            with tf.variable_scope("inception_5c"):
                end_points['inception_5c'] = get_inception_layer( end_points['inception_5b'], 128, 128, 256, 24, 64, 64 )

            with tf.variable_scope("inception_5d"):
                end_points['inception_5d'] = get_inception_layer( end_points['inception_5c'], 112, 144, 288, 32, 64, 64 )

            with tf.variable_scope("aux_logits_4"):
                end_points['aux_logits_4'] = aux_logit_layer( end_points['inception_5d'], num_classes, is_training )

            with tf.variable_scope("inception_5e"):
                end_points['inception_5e'] = get_inception_layer( end_points['inception_5d'], 256, 160, 320, 32, 128, 128 )

            end_points['pool4'] = layers.max_pool2d(end_points['inception_5e'], [ 3, 3 ], scope='pool4')

            #one tummy layer
            with tf.variable_scope("inception_6a"):
                end_points['inception_6a'] = get_inception_layer( end_points['pool4'], 192, 96, 208, 16, 48, 64 )

            with tf.variable_scope("aux_logits_5"):
                end_points['aux_logits_5'] = aux_logit_layer( end_points['inception_6a'], num_classes, is_training )

            with tf.variable_scope("inception_6b"):
                end_points['inception_6b'] = get_inception_layer( end_points['inception_6a'], 160, 112, 224, 24, 64, 64 )

            with tf.variable_scope("inception_6c"):
                end_points['inception_6c'] = get_inception_layer( end_points['inception_6b'], 128, 128, 256, 24, 64, 64 )

            with tf.variable_scope("inception_6d"):
                end_points['inception_6d'] = get_inception_layer( end_points['inception_6c'], 112, 144, 288, 32, 64, 64 )

            with tf.variable_scope("aux_logits_6"):
                end_points['aux_logits_6'] = aux_logit_layer( end_points['inception_6d'], num_classes, is_training )

            with tf.variable_scope("inception_6e"):
                end_points['inception_6e'] = get_inception_layer( end_points['inception_6d'], 256, 160, 320, 32, 128, 128 )

            end_points['pool5'] = layers.max_pool2d(end_points['inception_6e'], [ 3, 3 ], scope='pool5')

            
            with tf.variable_scope("inception_7a"):
                end_points['inception_7a'] = get_inception_layer( end_points['pool5'], 256, 160, 320, 32, 128, 128 )

            with tf.variable_scope("inception_7b"):
                end_points['inception_7b'] = get_inception_layer( end_points['inception_7a'], 384, 192, 384, 48, 128, 128 )
            print("before average: ", end_points['inception_7b'].get_shape().as_list())

            #end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [ 7, 7 ], stride = 1, scope='pool4')
            #end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [ 16, 16 ], stride = 1, scope='pool4')
            end_points['pool6'] = layers.avg_pool2d(end_points['inception_7b'], [ 4, 4 ], stride = 1, scope='pool6')
            #end_points['pool06'] = layers.avg_pool2d(end_points['inception_7b'], [ 2, 2 ], stride = 2, scope='pool06')
            #end_points['pool6'] = layers.avg_pool2d(end_points['pool06'], [ 2, 2 ], stride = 2, scope='pool6')
            
            pool4Shape = end_points['pool6'].get_shape().as_list()
            print (pool4Shape)


            end_points['reshape'] = tf.reshape( end_points['pool6'], [-1, pool4Shape[1] * pool4Shape[2] * pool4Shape[3]] )
            
            end_points['dropout'] = layers.dropout( end_points['reshape'], dropout_keep_prob, is_training = is_training )

            end_points['logits'] = layers.fully_connected( end_points['dropout'], 256, activation_fn=tf.nn.relu, scope='logits')
            
            end_points['logits'] = layers.dropout( end_points['logits'], dropout_keep_prob, is_training = is_training )

            end_points['logits1'] = layers.fully_connected( end_points['logits'], 16, activation_fn=tf.nn.relu, scope='logits1')
            
            end_points['logits2'] = layers.dropout( end_points['logits1'], dropout_keep_prob, is_training = is_training )
            
            end_points['logits3'] = layers.fully_connected( end_points['logits2'], num_classes, activation_fn=None, scope='logits3')

            end_points['predictions'] = tf.nn.softmax(end_points['logits3'], name='predictions')
            
            print("prediction: ", end_points['predictions'])
            print("prediction: ", end_points['logits3'])

    #return end_points['logits'], end_points
    #return end_points['predictions']
    return end_points['logits3']
'''

'''#The GoogleNet
print("the googlenet")
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops

def get_inception_layer( inputs, conv11_size, conv33_11_size, conv33_size,
                         conv55_11_size, conv55_size, pool11_size, is_training ):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d( inputs, conv11_size, [ 1, 1 ] )
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d( inputs, conv33_11_size, [ 1, 1 ] )
        conv33 = layers.conv2d( conv33_11, conv33_size, [ 3, 3 ] )
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d( inputs, conv55_11_size, [ 1, 1 ] )
        conv55 = layers.conv2d( conv55_11, conv55_size, [ 5, 5 ] )
    with tf.variable_scope("pool_proj"):
        pool_proj = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
        pool11 = layers.conv2d( pool_proj, pool11_size, [ 1, 1 ] )
    if tf.__version__ == '0.11.0rc0':
        before_norm_tf = tf.concat(3, [conv11, conv33, conv55, pool11])
        before_norm_tf = layers.batch_norm(inputs=before_norm_tf, is_training=is_training )
    else:
        before_norm_tf = tf.concat([conv11, conv33, conv55, pool11], 3)
        before_norm_tf = layers.batch_norm(inputs=before_norm_tf, is_training=is_training )
    return before_norm_tf

def aux_logit_layer( inputs, num_classes, is_training ):
    with tf.variable_scope("pool2d"):
        pooled = layers.avg_pool2d(inputs, [ 5, 5 ], stride = 3 )
    with tf.variable_scope("conv11"):
        conv11 = layers.conv2d( pooled, 128, [1, 1] )
    temp_list  = conv11.get_shape().as_list()
    with tf.variable_scope("flatten"):
        #flat = tf.reshape( conv11, [-1, 2048] )
        flat = tf.reshape( conv11, [-1, temp_list[1]*temp_list[2]*temp_list[3]] )
    with tf.variable_scope("fc"):
        fc = layers.fully_connected( flat, 1024, activation_fn=None )
        #fc = layers.fully_connected( flat, 256, activation_fn=None )
    with tf.variable_scope("drop"):
        drop = layers.dropout( fc, 0.3, is_training = is_training )
    with tf.variable_scope( "linear" ):
        linear = layers.fully_connected( drop, num_classes, activation_fn=None )
    with tf.variable_scope("soft"):
        soft = tf.nn.softmax( linear )
    return soft

def googlenet(inputs,
              dropout_keep_prob=0.4,
              is_training=True,scope=''):

    
    #Implementation of https://arxiv.org/pdf/1409.4842.pdf
    
    num_classes=3
    end_points = {}
    with tf.name_scope( scope, "googlenet", [inputs] ):
        with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
            end_points['conv0'] = layers.conv2d( inputs, 64, [ 7, 7 ], stride = 2, scope = 'conv0' )
            print("conv0", end_points['conv0'].get_shape().as_list())
            
            end_points['pool0'] = layers.max_pool2d(end_points['conv0'], [3, 3], scope='pool0')
            print("pool0",end_points['pool0'].get_shape().as_list())
            
            end_points['conv1_a'] = layers.conv2d( end_points['pool0'], 64, [ 1, 1 ], scope = 'conv1_a' )
            print("conv1_a",end_points['conv1_a'].get_shape().as_list())
            
            end_points['conv1_b'] = layers.conv2d( end_points['conv1_a'], 192, [ 3, 3 ], scope = 'conv1_b' )
            print("conv1_b", end_points['conv1_b'].get_shape().as_list())
            
            end_points['pool1'] = layers.max_pool2d(end_points['conv1_b'], [ 3, 3 ], scope='pool1')
            print("pool1",end_points['pool1'].get_shape().as_list())
            
            #end_points['pool1'] = layers.batch_norm(inputs=end_points['pool1'], is_training=is_training )

            with tf.variable_scope("inception_3a"):
                end_points['inception_3a'] = get_inception_layer( end_points['pool1'], 64, 96, 128, 16, 32, 32, is_training )
                print("inception_3a",end_points['inception_3a'].get_shape().as_list())

            with tf.variable_scope("inception_3b"):
                end_points['inception_3b'] = get_inception_layer( end_points['inception_3a'], 128, 128, 192, 32, 96, 64, is_training )
                print("inception_3b",end_points['inception_3b'].get_shape().as_list())

            end_points['pool2'] = layers.max_pool2d(end_points['inception_3b'], [ 3, 3 ], scope='pool2')
            print("pool2",end_points['pool2'].get_shape().as_list())
            
            #end_points['pool2'] = layers.batch_norm(inputs=end_points['pool2'], is_training=is_training )

            with tf.variable_scope("inception_4a"):
                end_points['inception_4a'] = get_inception_layer( end_points['pool2'], 192, 96, 208, 16, 48, 64, is_training )
                print("inception_4a",end_points['inception_4a'].get_shape().as_list())

            with tf.variable_scope("aux_logits_1"):
                end_points['aux_logits_1'] = aux_logit_layer( end_points['inception_4a'], num_classes, is_training )
                print("aux_logits_1",end_points['aux_logits_1'].get_shape().as_list())

            with tf.variable_scope("inception_4b"):
                end_points['inception_4b'] = get_inception_layer( end_points['inception_4a'], 160, 112, 224, 24, 64, 64, is_training )
                print("inception_4b",end_points['inception_4b'].get_shape().as_list())

            with tf.variable_scope("inception_4c"):
                end_points['inception_4c'] = get_inception_layer( end_points['inception_4b'], 128, 128, 256, 24, 64, 64, is_training )
                print("inception_4c",end_points['inception_4c'].get_shape().as_list())

            with tf.variable_scope("inception_4d"):
                end_points['inception_4d'] = get_inception_layer( end_points['inception_4c'], 112, 144, 288, 32, 64, 64, is_training )
                print("inception_4d",end_points['inception_4d'].get_shape().as_list())

            with tf.variable_scope("aux_logits_2"):
                end_points['aux_logits_2'] = aux_logit_layer( end_points['inception_4d'], num_classes, is_training )
                print("aux_logits_2",end_points['aux_logits_2'].get_shape().as_list())

            with tf.variable_scope("inception_4e"):
                end_points['inception_4e'] = get_inception_layer( end_points['inception_4d'], 256, 160, 320, 32, 128, 128, is_training )
                print("inception_4e",end_points['inception_4e'].get_shape().as_list())

            end_points['pool3'] = layers.max_pool2d(end_points['inception_4e'], [ 3, 3 ], scope='pool3')
            print("pool3",end_points['pool3'].get_shape().as_list())
            #end_points['pool3'] = layers.batch_norm(inputs=end_points['pool3'], is_training=is_training )

            with tf.variable_scope("inception_5a"):
                end_points['inception_5a'] = get_inception_layer( end_points['pool3'], 256, 160, 320, 32, 128, 128, is_training )
                print("inception_5a",end_points['inception_5a'].get_shape().as_list())

            with tf.variable_scope("inception_5b"):
                end_points['inception_5b'] = get_inception_layer( end_points['inception_5a'], 384, 192, 384, 48, 128, 128, is_training )
                print("inception_5b",end_points['inception_5b'].get_shape().as_list())

            end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [ 7, 7 ], stride = 1, scope='pool4')
            #end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [ 16, 16 ], stride = 1, scope='pool4')
            print("pool4",end_points['pool4'].get_shape().as_list())

            pool4Shape = end_points['pool4'].get_shape().as_list()
            end_points['reshape'] = tf.reshape( end_points['pool4'], [-1, pool4Shape[1] * pool4Shape[2] * pool4Shape[3]] )
            print("reshape",end_points['reshape'].get_shape().as_list())
            
            end_points['dropout'] = layers.dropout( end_points['reshape'], dropout_keep_prob, is_training = is_training )
            print("dropout",end_points['dropout'].get_shape().as_list())

            end_points['logits'] = layers.fully_connected( end_points['dropout'], num_classes, activation_fn=None, scope='logits')
            print("logits",end_points['logits'].get_shape().as_list())

            end_points['predictions'] = tf.nn.softmax(end_points['logits'], name='predictions')
            print("predictions",end_points['predictions'].get_shape().as_list())

    #return end_points['logits'], end_points
    #return end_points['logits']
    return end_points['logits'], end_points['aux_logits_1'] , end_points['aux_logits_2']
'''
'''import tensorflow as tf
print ("old implementation")
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops

def get_inception_layer( inputs, conv11_size, conv33_11_size, conv33_size,
                         conv55_11_size, conv55_size, pool11_size ):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d( inputs, conv11_size, [ 1, 1 ] )
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d( inputs, conv33_11_size, [ 1, 1 ] )
        conv33 = layers.conv2d( conv33_11, conv33_size, [ 3, 3 ] )
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d( inputs, conv55_11_size, [ 1, 1 ] )
        conv55 = layers.conv2d( conv55_11, conv55_size, [ 5, 5 ] )
    with tf.variable_scope("pool_proj"):
        pool_proj = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
        pool11 = layers.conv2d( pool_proj, pool11_size, [ 1, 1 ] )
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool11])
    return tf.concat([conv11, conv33, conv55, pool11], 3)

def aux_logit_layer( inputs, num_classes, is_training ):
    with tf.variable_scope("pool2d"):
        pooled = layers.avg_pool2d(inputs, [ 5, 5 ], stride = 3 )
    with tf.variable_scope("conv11"):
        conv11 = layers.conv2d( pooled, 48, [1, 1] )
    with tf.variable_scope("flatten"):
        cov11Shape = conv11.get_shape().as_list()
        flat = tf.reshape( conv11, [-1, (cov11Shape[1] * cov11Shape[2] * cov11Shape[3])] )
    with tf.variable_scope("fc"):
        fc = layers.fully_connected( flat, 48, activation_fn=None )
    with tf.variable_scope("drop"):
        drop = layers.dropout( fc, 0.3, is_training = is_training )
    with tf.variable_scope( "linear" ):
        linear = layers.fully_connected( drop, num_classes, activation_fn=None )
    with tf.variable_scope("soft"):
        soft = tf.nn.softmax( linear )
    return soft

def googlenet(inputs,
              dropout_keep_prob=0.4,
              num_classes=1000,
              is_training=True,
              restore_logits = None,
              scope=''):
    #Implementation of https://arxiv.org/pdf/1409.4842.pdf

    end_points = {}
    with tf.name_scope( scope, "googlenet", [inputs] ):
        with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
            end_points['conv0'] = layers.conv2d( inputs, 4, [ 7, 7 ], stride = 2, scope = 'conv0' )
            print("conv0", end_points['conv0'].get_shape().as_list())
            
            end_points['pool0'] = layers.max_pool2d(end_points['conv0'], [3, 3], scope='pool0')
            print("pool0",end_points['pool0'].get_shape().as_list())
            
            end_points['conv1_a'] = layers.conv2d( end_points['pool0'], 4, [ 1, 1 ], scope = 'conv1_a' )
            print("conv1_a",end_points['conv1_a'].get_shape().as_list())

            end_points['conv1_b'] = layers.conv2d( end_points['conv1_a'], 12, [ 3, 3 ], scope = 'conv1_b' )
            print("conv1_b",end_points['conv1_b'].get_shape().as_list())

            end_points['pool1'] = layers.max_pool2d(end_points['conv1_b'], [ 3, 3 ], scope='pool1')
            print("pool1",end_points['pool1'].get_shape().as_list())


            with tf.variable_scope("inception_3a"):
                end_points['inception_3a'] = get_inception_layer( end_points['pool1'], 2, 2, 6, 2, 2, 2 )
                print("inception_3a",end_points['inception_3a'].get_shape().as_list())


            with tf.variable_scope("inception_3b"):
                end_points['inception_3b'] = get_inception_layer( end_points['inception_3a'], 4, 4, 12, 4, 4, 4 )
                print("inception_3b",end_points['inception_3b'].get_shape().as_list())


            end_points['pool2'] = layers.max_pool2d(end_points['inception_3b'], [ 3, 3 ], scope='pool2')
            print("pool2",end_points['pool2'].get_shape().as_list())


            with tf.variable_scope("inception_4a"):
                end_points['inception_4a'] = get_inception_layer( end_points['pool2'], 2, 2, 6, 2, 2, 2 )
                print("inception_4a",end_points['inception_4a'].get_shape().as_list())


            with tf.variable_scope("aux_logits_1"):
                end_points['aux_logits_1'] = aux_logit_layer( end_points['inception_4a'], num_classes, is_training )
                print("aux_logits_1",end_points['aux_logits_1'].get_shape().as_list())


            with tf.variable_scope("inception_4b"):
                end_points['inception_4b'] = get_inception_layer( end_points['inception_4a'], 2, 2, 6, 2, 2, 2 )
                print("inception_4b",end_points['inception_4b'].get_shape().as_list())


            with tf.variable_scope("inception_4c"):
                end_points['inception_4c'] = get_inception_layer( end_points['inception_4b'], 4, 4, 12, 4, 4, 4 )
                print("inception_4c",end_points['inception_4c'].get_shape().as_list())

            with tf.variable_scope("inception_4d"):
                end_points['inception_4d'] = get_inception_layer( end_points['inception_4c'], 4, 4, 12, 4, 4, 4 )
                print("inception_4d",end_points['inception_4d'].get_shape().as_list())


            with tf.variable_scope("aux_logits_2"):
                end_points['aux_logits_2'] = aux_logit_layer( end_points['inception_4d'], num_classes, is_training )
                print("aux_logits_2",end_points['aux_logits_2'].get_shape().as_list())


            with tf.variable_scope("inception_4e"):
                end_points['inception_4e'] = get_inception_layer( end_points['inception_4d'], 10, 6, 14, 6, 6, 6 )
                print("inception_4e",end_points['inception_4e'].get_shape().as_list())


            end_points['pool3'] = layers.max_pool2d(end_points['inception_4e'], [ 3, 3 ], scope='pool3')
            print("pool3",end_points['pool3'].get_shape().as_list())


            with tf.variable_scope("inception_5a"):
                end_points['inception_5a'] = get_inception_layer( end_points['pool3'], 4, 4, 12, 4, 4, 4 )
                print("inception_5a",end_points['inception_5a'].get_shape().as_list())


            with tf.variable_scope("inception_5b"):
                end_points['inception_5b'] = get_inception_layer( end_points['inception_5a'], 8, 8, 24, 8, 8, 8 )
                print("inception_5b",end_points['inception_5b'].get_shape().as_list())


            end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [ 7, 7 ], stride = 1, scope='pool4')
            #end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [ 16, 16 ], stride = 1, scope='pool4')
            print("pool4",end_points['pool4'].get_shape().as_list())


            #end_points['reshape'] = tf.reshape( end_points['pool4'], [-1, 1024] )
            pool4Shape = end_points['pool4'].get_shape().as_list()
            end_points['reshape'] = tf.reshape( end_points['pool4'], [-1, pool4Shape[1] * pool4Shape[2] * pool4Shape[3]] )
            print("reshape",end_points['reshape'].get_shape().as_list())

                        
            end_points['dropout'] = layers.dropout( end_points['reshape'], dropout_keep_prob, is_training = is_training )
            end_points['logits_1'] = layers.fully_connected( end_points['dropout'], 1000, activation_fn=None, scope='logits_1')
            print("logits_1",end_points['logits_1'].get_shape().as_list())

            
            end_points['dropout_1'] = layers.dropout( end_points['logits_1'], dropout_keep_prob, is_training = is_training )
            end_points['logits_2'] = layers.fully_connected( end_points['dropout_1'], 500, activation_fn=None, scope='logits_2')
            print("logits_2",end_points['logits_2'].get_shape().as_list())

            
            end_points['dropout_2'] = layers.dropout( end_points['logits_2'], dropout_keep_prob, is_training = is_training )
            end_points['logits_3'] = layers.fully_connected( end_points['dropout_2'], 250, activation_fn=None, scope='logits_3')
            print("logits_3",end_points['logits_3'].get_shape().as_list())

            
            end_points['dropout_3'] = layers.dropout( end_points['logits_3'], dropout_keep_prob, is_training = is_training )
            end_points['logits_4'] = layers.fully_connected( end_points['dropout_3'], 125, activation_fn=None, scope='logits_4')
            print("logits_4",end_points['logits_4'].get_shape().as_list())

            
            end_points['dropout_4'] = layers.dropout( end_points['logits_4'], dropout_keep_prob, is_training = is_training )
            end_points['logits_5'] = layers.fully_connected( end_points['dropout_4'], 64, activation_fn=None, scope='logits_5')
            print("logits_5",end_points['logits_5'].get_shape().as_list())

            
            end_points['dropout_5'] = layers.dropout( end_points['logits_5'], dropout_keep_prob, is_training = is_training )
            end_points['logits_6'] = layers.fully_connected( end_points['dropout_5'], 3, activation_fn=None, scope='logits_6')
            print("logits_6",end_points['logits_6'].get_shape().as_list())


            end_points['predictions'] = tf.nn.softmax(end_points['logits_6'], name='predictions')
            print("predictions",end_points['predictions'].get_shape().as_list())


    return end_points['logits_6'], end_points['aux_logits_1'] , end_points['aux_logits_2']
'''





from sklearn.preprocessing import LabelBinarizer
def display_image_predictions(features, labels, predictions):
    n_classes = 3
    label_names = ['meningioma', 'glioma', 'pituitary']
    '''label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))'''

    fig, axies = plt.subplots(nrows=3, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions
    

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, labels, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]

        #print (np.array(feature).shape, correct_name)
        test_1_image = np.reshape(feature,(512,512))
        test_2_image = np.array(test_1_image,dtype=float)
        axies[image_i][0].imshow(test_2_image)

        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()
        
        #print(pred_names, pred_values)
        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
    plt.show() 
        
        
def batch_features_labels(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]
        
        

def test_model(save_model_path, test_features, test_labels):
    batch_size = 64
    n_samples = 3
    top_n_predictions = 3

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        #loaded_logits = loaded_graph.get_tensor_by_name('logits:0') 
        loaded_logits = tf.get_collection('logits')[0]
        
        #loaded_logits = loaded_graph.get_tensor_by_name('aux_logits_2:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        loaded_is_training = loaded_graph.get_tensor_by_name('is_training:0')

        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, \
                           loaded_keep_prob: 1.0, loaded_is_training: False})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
        print ('Number of test batches : {}\n'.format(test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0,loaded_is_training: False})
        
        display_image_predictions (random_test_features, random_test_labels, random_test_predictions)
        #return [random_test_features, random_test_labels, random_test_predictions]
        