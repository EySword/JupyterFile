import tensorflow as tf
import numpy as np
INPUT_NODE = 30
OUTPUT_NODE = 1
LAYER1_NODE = 60

def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1=get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1=get_bias([LAYER1_NODE])
    y1=tf.nn.tanh(tf.matmul(x,w1)+b1)

    w2=get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
    b2=get_bias([OUTPUT_NODE])
    y=tf.matmul(y1,w2)+b2
    return y

