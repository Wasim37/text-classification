import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import shutil
import time


time0 = time.time()
print('Starting ...')
model_name = 'p3-2-cnn-256-2357'                    # 模型名称
W_embedding = np.load('../data/word_embedding.npy').astype(np.float32)            # 导入预训练好的词向量
# W_topic_embedding = np.load('../data/W_topic_embedding.npy').astype(np.float32)
model_path = '../ckpt/' + model_name + '/'                  # 模型保存位置
summary_path = '../summary/' + model_name + '/'             # summary 位置
result_path = '../result/' + model_name + '.csv'            # result.csv 位置
scores_path = '../scores/' + model_name + '.npy'            # scores.npy 位置
local_scores_path = '../local_scores/' + model_name + '.npy'


if not os.path.exists(model_path):
    os.makedirs(model_path)         
model_path = model_path + 'model.ckpt'
if os.path.exists(summary_path):   # 删除原来的 summary 文件，避免重合
    print('removed the existing summary files.')
    shutil.rmtree(summary_path)
os.makedirs(summary_path)          # 然后再次创建
    
# ##################### config ######################
n_step1 = max_len1 = 30                   # title句子长度
n_step2= max_len2 = 150                   # content 长度
input_size = embedding_size = 256       # 字向量长度
n_class = 1999                          # 类别总数
filter_sizes = [2,3,5,7]                  # 卷积核大小
n_filter = 256                          # 每种卷积核的个数
fc_hidden_size = 1024                   # fc 层节点数
n_filter_total = n_filter * len(filter_sizes)
valid_num = 100000
seed_num = 13
tr_batch_size = 128
te_batch_size = 128
print('Prepared, costed time %g s.' % (time.time() - time0))









import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

'''
双端 GRU，知乎问题多标签分类。
'''
print('Building model ...')
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
tst = tf.placeholder(tf.bool)
n_updates = tf.placeholder(tf.int32)      # training iteration,传入 bn 层
update_emas = list()                       # BN 层中所有的更新操作


def weight_variable(shape, name, initializer=None):
    """Create a weight variable with appropriate initialization."""
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32)

def bias_variable(shape, name):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name=name, initializer=initial, dtype=tf.float32)

def batchnorm(Ylogits, is_test, num_updates, offset, convolutional=False):
    """batchnormalization.
    Args:
        Ylogits: 1D向量或者是3D的卷积结果。
        num_updates: 迭代的global_step
        offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
        scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
        m: 表示batch均值；v:表示batch方差。
        bnepsilon：一个很小的浮点数，防止除以 0.
    Returns:
        Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
        update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
    """
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, num_updates) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages


with tf.name_scope('Inputs'):
    X1_inputs = tf.placeholder(tf.int64, [None, n_step1], name='X1_input')
    X2_inputs = tf.placeholder(tf.int64, [None, n_step2], name='X2_input')
    y_inputs = tf.placeholder(tf.float32, [None, n_class], name='y_input')    

with tf.device('/cpu:0'):
    with tf.variable_scope('embedding') as vs:
        embedding = tf.get_variable(name="W_embedding", shape=W_embedding.shape, 
                            initializer=tf.constant_initializer(W_embedding), trainable=True)   # fine-tune

def textcnn(X_inputs, n_step):
    """build the TextCNN network. Return the h_drop"""
    # X_inputs.shape = [batchsize, n_step]  ->  inputs.shape = [batchsize, n_step, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)  
    inputs = tf.expand_dims(inputs, -1)
    pooled_outputs = list()
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, n_filter]
            W_filter = weight_variable(shape=filter_shape, name='W_filter')
            beta = bias_variable(shape=[n_filter], name='beta_filter')
            tf.summary.histogram('beta_filter', beta)
            conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            conv_bn, update_ema = batchnorm(conv, tst, n_updates, beta, convolutional=True)    # 在激活层前面加 BN
            # Apply nonlinearity, batch norm scaling is not useful with relus
            # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
            h = tf.nn.relu(conv_bn, name="filter_relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,ksize=[1, n_step - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],padding='VALID',name="pool")
            pooled_outputs.append(pooled)
            update_emas.append(update_ema)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, n_filter_total]) 
    return h_pool_flat    # shape = [-1, n_filter_total]
    
    
with tf.variable_scope('cnn-title'):
    output_title = textcnn(X1_inputs, n_step1)
with tf.variable_scope('cnn-content'):
    output_content = textcnn(X2_inputs, n_step2)
with tf.variable_scope('fc-bn-layer'):
    output = tf.concat([output_title, output_content], axis=1)
    W_fc = weight_variable([n_filter_total*2, fc_hidden_size], name='Weight_fc')
    tf.summary.histogram('W_fc', W_fc)
    h_fc = tf.matmul(output, W_fc, name='h_fc')
    beta_fc = bias_variable([fc_hidden_size], name="beta_fc")
    tf.summary.histogram('beta_fc', beta_fc)
    fc_bn, update_ema_fc = batchnorm(h_fc, tst, n_updates, beta_fc, convolutional=False)
    update_emas.append(update_ema_fc)
    fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
    fc_bn_drop = tf.nn.dropout(fc_bn_relu, keep_prob, name="fc_dropout")

with tf.variable_scope('out_layer'):
    W_out = weight_variable(shape=[fc_hidden_size, n_class], name='Weight_out')
    tf.summary.histogram('Weight_out', W_out)
    b_out = bias_variable([n_class], name='bias_out') 
    tf.summary.histogram('bias_out', b_out)
    y_pred = tf.nn.xw_plus_b(fc_bn_drop, W_out, b_out, name='y_pred')  #每个类别的分数 scores
    
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_inputs))
    tf.summary.scalar('cost', cost)

#  ------------- 优化器设置 ---------------------
global_step = tf.Variable(0, trainable=False, name='Global_Step')
update_global_step = tf.assign(global_step, global_step+1)
starter_learning_rate = 5.0e-4
decay_step = 10000
# decay_step = 150  # 测试用
decay = 0.90 
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay, staircase=False)
with tf.variable_scope('AdamOptimizer1'):
    tvars1 = tf.trainable_variables()
    grads1 = tf.gradients(cost, tvars1)
    optimizer1 = tf.train.AdamOptimizer(learning_rate)
    train_op1 = optimizer1.apply_gradients(zip(grads1, tvars1),
        global_step=global_step)

with tf.variable_scope('AdamOptimizer2'):          # 只更新新创建的部分变量
    tvars2 = [tvar for tvar in tvars1 if 'embedding' not in tvar.name]
    grads2 = tf.gradients(cost, tvars2)
    optimizer2 = tf.train.AdamOptimizer(learning_rate)
    train_op2 = optimizer2.apply_gradients(zip(grads2, tvars2),
        global_step=global_step)
update_op = tf.group(*update_emas)   # 更新 BN 参数    

# summary
merged = tf.summary.merge_all() # summary
train_writer = tf.summary.FileWriter(summary_path + 'train', sess.graph)
test_writer = tf.summary.FileWriter(summary_path + 'test')
print('Finished creating the TextCNN model.')










sys.path.append('..')
from data_helpers import BatchGenerator
from data_helpers import to_categorical
from evaluator import score_eval


# data_train_path = '/home/huangyongye/zhihu_data/data_train/'
# data_valid_path = '/home/huangyongye/zhihu_data/data_valid/'
data_train_path = '../data/wd-data/data_train/'
data_valid_path = '../data/wd-data/data_valid/'
tr_batches = os.listdir(data_train_path)   # batch 文件名列表
va_batches = os.listdir(data_valid_path)
n_tr_batches = len(tr_batches)
n_va_batches = len(va_batches)


def get_batch(data_path, batch_id, title_len=n_step1):
    """get a batch from data_path"""
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    X1_batch = X_batch[:, :title_len]
    X2_batch = X_batch[:, title_len:]
    return [X1_batch, X2_batch, y_batch]


def valid_epoch(data_path=data_valid_path):
    """Test on the valid data."""
    _costs = 0.0
    predict_labels_list = list()  # 所有的预测结果
    marked_labels_list = list()   # 真实标签
    _global_step = sess.run(global_step)
    for i in range(n_va_batches):
        [X1_batch, X2_batch, y_batch] = get_batch(data_path, i)
        marked_labels_list.extend(y_batch)
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        fetches = [merged, cost, y_pred]  
        feed_dict = {X1_inputs:X1_batch, X2_inputs:X2_batch,  y_inputs:y_batch, 
                     batch_size:_batch_size, keep_prob:1.0, tst:True, n_updates:_global_step}
        summary, _cost, predict_labels = sess.run(fetches, feed_dict)
        _costs += _cost
        predict_labels = map(lambda label: label.argsort()[-1:-6:-1], predict_labels) # 取最大的5个下标
        predict_labels_list.extend(predict_labels)
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)
    mean_cost = _costs / n_va_batches
    return mean_cost, precision, recall, f1

print('Every thing prepared!')




# 测试
valid_step = 10000        # 每 valid_step 就进行一次 valid 运算
max_epoch = 4           # cpu迭代次数
max_max_epoch = 7       # 最多迭代的次数

print('tr_batch_num=%d' % n_tr_batches)
saver = tf.train.Saver(max_to_keep=5)           # 最多保存的模型数量
sess.run(tf.global_variables_initializer())
last_f1 = 0.40
model_num = 0







time0 = time.time()
train_op = train_op2 

for epoch in range(max_max_epoch):
    batch_indexs = np.random.permutation(n_tr_batches)  # shuffle the training data
    if epoch == max_epoch:
        train_op = train_op1
        if model_num == 0:
            model_num += 1
            save_path = saver.save(sess, model_path, global_step=model_num)
            print('the save path is ', save_path)
        print('Begin updating embedding.')
    print('EPOCH %d, lr= %g' % (epoch+1, sess.run(learning_rate)))    
    for batch in range(n_tr_batches): 
        _global_step = sess.run(global_step)
        if (_global_step+1) % valid_step == 0:    # 进行 valid 计算
            valid_cost, precision, recall, f1 = valid_epoch()
            print('Global_step=%d: valid cost=%g; p=%g, r=%g, f1=%g, time=%g s' % (
                    _global_step, valid_cost, precision, recall, f1, time.time()-time0))
            time0 = time.time()
            if (f1 > last_f1):
                last_f1 = f1
                model_num += 1
                save_path = saver.save(sess, model_path, global_step=model_num)
                print('the save path is ', save_path) 
                
        batch_id = batch_indexs[batch]
        [X1_batch, X2_batch, y_batch] = get_batch(data_train_path, batch_id, n_step1)
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        fetches = [merged, cost, train_op, update_op]
        feed_dict = {X1_inputs:X1_batch, X2_inputs:X2_batch, y_inputs:y_batch, batch_size:_batch_size, 
                     keep_prob:0.5, tst:False, n_updates:_global_step}
        summary, _cost, _, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
        if _global_step % 100:
            train_writer.add_summary(summary, _global_step)
            batch_id = np.random.randint(0, n_va_batches)   # 随机选一个验证batch
            [X1_batch, X2_batch, y_batch] = get_batch(data_valid_path, batch_id, n_step1)
            y_batch = to_categorical(y_batch)
            _batch_size = len(y_batch)
            feed_dict = {X1_inputs:X1_batch, X2_inputs:X2_batch,  y_inputs:y_batch,
                         batch_size:_batch_size, keep_prob:1.0, tst:True, n_updates:_global_step}
            fetches = [merged, cost]
            summary, _cost = sess.run(fetches, feed_dict)
            test_writer.add_summary(summary, _global_step)
        valid_cost, precision, recall, f1 = valid_epoch()  # # 每个 epoch 进行一次验证 valid     

valid_cost, precision, recall, f1 = valid_epoch()  # # 每个 epoch 进行一次验证 valid
print('Global_step=%d;  valid cost=%g; p=%g, r=%g, f1=%g; speed=%g s/epoch' % (
    _global_step, valid_cost, precision, recall, f1, time.time()-time0) )
if (f1 > last_f1):
    model_num += 1
    save_path = saver.save(sess, model_path, global_step=model_num)
    print('the save path is ', save_path) 