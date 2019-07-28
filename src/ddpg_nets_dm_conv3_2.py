# Heavily modified

import numpy as np
import tensorflow as tf


def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)


def theta_p(dimO, dimA, l1, l2):
    dimO = dimO[0]
    dimA = dimA[0]
    with tf.variable_scope("theta_p"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                tf.Variable(tf.random_uniform([l2, dimA], -3e-3, 3e-3), name='3w'),
                tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='3b')]


def policy(obs, theta, name='policy'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
        h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
        action = tf.nn.tanh(h3, name='h4-action')
        return action


def theta_q(dimO, dimA, l1, l2):
    #l1= l2 = 100
    dimO = dimO[0]
    dimA = dimA[0]
    with tf.variable_scope("theta_q"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w_o'),
                tf.Variable(fanin_init([l1], dimO), name='1b_o'),
                tf.Variable(fanin_init([dimA, l1]), name='1w_a'),
                tf.Variable(fanin_init([l1], dimO), name='1b_a'),
                tf.Variable(fanin_init([l1, l2]), name='2w_u'),
                tf.Variable(fanin_init([l2], dimO), name='2b_u'),
                tf.Variable(fanin_init([l1, dimA]), name='2w_yu'),
                tf.Variable(fanin_init([dimA], dimO), name='2b_yu'),
                tf.Variable(fanin_init([dimA, l2]), name='2w_a'),
                tf.Variable(fanin_init([l1,l1]), name='2w_zu'),
                tf.Variable(fanin_init([l1], dimO), name='2b_zu'),
                tf.Variable(fanin_init([l1, l2]), name='2w_conv'),
                tf.Variable(fanin_init([l1, l2]), name='3w_u'),
                tf.Variable(fanin_init([l2], dimO), name='3b_u'),
                tf.Variable(fanin_init([l1, dimA]), name='3w_yu'),
                tf.Variable(fanin_init([dimA], dimO), name='3b_yu'),
                tf.Variable(fanin_init([dimA, l2]), name='3w_a'),
                tf.Variable(fanin_init([l1,l1]), name='3w_zu'),
                tf.Variable(fanin_init([l1], dimO), name='3b_zu'),
                tf.Variable(fanin_init([l1, l2]), name='3w_conv'),
                tf.Variable(tf.random_uniform([l2, 1], -3e-4, 3e-4), name='4w_u'),
                tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name='4b_u'),
                tf.Variable(fanin_init([l2, dimA]), name='4w_yu'),
                tf.Variable(fanin_init([dimA], dimO), name='4b_yu'),
                tf.Variable(tf.random_uniform([dimA, 1],-3e-4, 3e-4), name='4w_a'),
                tf.Variable(fanin_init([l2,l2]), name='4w_zu'),
                tf.Variable(fanin_init([l2], dimO), name='4b_zu'),
                tf.Variable(tf.random_uniform([l2, 1],-3e-4, 3e-4), name='4w_conv')]

def qfunction(obs, act, theta, name="qfunction"):
    with tf.variable_op_scope([obs, act], name, name):
        h0_o = tf.identity(obs, name='h0-obs')
        h0_a = tf.identity(act, name='h0-act')
        h1_o = tf.matmul(h0_o, theta[0]) + theta[1]
        h1_a = tf.matmul(h0_a,theta[2]) + theta[3]
        h1 = tf.nn.relu(h1_o+h1_a, name='h1')
        h2_u = tf.matmul(h1_o, theta[4]) + theta[5]
        h2_a = tf.matmul(tf.multiply(h0_a,tf.matmul(h1_o,theta[6])+theta[7]),theta[8])
        h2_z = tf.matmul(tf.multiply(h1,tf.matmul(h1_o,theta[9])+theta[10]),theta[11])
        h2 = tf.nn.relu(h2_u+ h2_a + h2_z)
        h3_u = tf.matmul(h2_u, theta[12]) + theta[13]
        h3_a = tf.matmul(tf.multiply(h0_a,tf.matmul(h2_u,theta[14])+theta[15]),theta[16])
        h3_z = tf.matmul(tf.multiply(h2,tf.matmul(h2_u,theta[17])+theta[18]),theta[19])
        h3 = tf.nn.relu(h3_u+ h3_a + h3_z)
        h4_u = tf.matmul(h3_u, theta[20]) + theta[21]
        h4_a = tf.matmul(tf.multiply(h0_a,tf.matmul(h3_u,theta[22])+theta[23]),theta[24])
        h4_z = tf.matmul(tf.multiply(h3,tf.matmul(h3_u,theta[25])+theta[26]),theta[27])
        qs = h4_u + h4_a + h4_z
        q = tf.squeeze(qs, [1], name='h3-q')
        return q
