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
model_name = 'p2-1-rnn-cnn-256-256'                    # 模型名称
W_embedding = np.load('../data/word_embedding.npy')            # 导入预训练好的词向量
model_path = '../ckpt/' + model_name + '/'                  # 模型保存位置
summary_path = '../summary/' + model_name + '/'             # summary 位置
result_path = '../result/' + model_name + '.csv'            # result.csv 位置
scores_path = '../scores/' + model_name + '.npy'            # scores.npy 位置
save_path = '../data/'

if not os.path.exists(model_path):
    os.makedirs(model_path)         
model_path = model_path + 'model.ckpt'
if os.path.exists(summary_path):   # 删除原来的 summary 文件，避免重合
    shutil.rmtree(summary_path)
os.makedirs(summary_path)          # 然后再次创建


best_model_path = '../ckpt/p1-1-bigru-512/model.ckpt-6'
# ##################### config ######################
n_step1 = max_len1 = 50                   # title句子长度
n_step2= max_len2 = 150                   # content 长度
input_size = embedding_size = 256       # 字向量长度
hidden_size = 256    # 隐含层节点数
n_layer = 1        # bi-gru 层数
fc_hidden_size = 1024                   # fc 层节点数
filter_sizes = [2,3,4,5,7]                  # 卷积核大小
n_filter = 256                             # 每种卷积核的个数512

# 测试参数
# fc_hidden_size = 1024                   # fc 层节点数
# filter_sizes = [3,4,5]                  # 卷积核大小
# n_filter = 100                             # 每种卷积核的个数512


n_filter_total = n_filter * len(filter_sizes)
n_class = 1999

max_grad_norm = 1.0  # 最大梯度（超过此值的梯度将被裁剪）
global_step = 0
valid_num = 100000
seed_num = 13
te_batch_size = 128 
tr_batch_size = 128 
print('Prepared, costed time %g s.' % (time.time() - time0))


#---------------- 创建模型

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

'''
HAN-CNN-BiGRU 模型，知乎问题多标签分类。
'''

print('Building model ...')
lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
tst = tf.placeholder(tf.bool)
n_updates = tf.placeholder(tf.int32)      # training iteration,传入 bn 层
update_emas = list()   # BN 层中所有的更新操作


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


# 第一次定义，放在 CPU 上面
with tf.device('/cpu:0'):
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable(name="W_embedding", shape=W_embedding.shape, 
                        initializer=tf.constant_initializer(W_embedding), trainable=True)   # fine-tune

with tf.name_scope('Inputs'):
    X1_inputs = tf.placeholder(tf.int64, [None, n_step1], name='X1_input')
    X2_inputs = tf.placeholder(tf.int64, [None, n_step2], name='X2_input')
    y_inputs = tf.placeholder(tf.float32, [None, n_class], name='y_input')
    inputs1 = tf.nn.embedding_lookup(embedding, X1_inputs)
    inputs2 = tf.nn.embedding_lookup(embedding, X2_inputs) 


def gru_cell():
    with tf.name_scope('gru_cell'):
        cell = rnn.GRUCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def bi_gru(inputs):
    """build the bi-GRU network. 返回个最后一层的隐含状态。"""      
    cells_fw = [gru_cell() for _ in range(n_layer)]
    cells_bw = [gru_cell() for _ in range(n_layer)]
    initial_states_fw = [cell_fw.zero_state(batch_size, tf.float32) for cell_fw in cells_fw]
    initial_states_bw = [cell_bw.zero_state(batch_size, tf.float32) for cell_bw in cells_bw] 
    outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, 
                        initial_states_fw = initial_states_fw, initial_states_bw = initial_states_bw, dtype=tf.float32)
    return outputs
    

# 这部分数据需要进行保存
with tf.variable_scope('bigru_title'):          # 两部分的输出都加上 BN+RELU 层，
    word_encoder_title = bi_gru(inputs1)      # title 部分输出
    
with tf.variable_scope('bigru_content'):
    word_encoder_content = bi_gru(inputs2)    # content 部分输出
    
saver = tf.train.Saver(max_to_keep=3)           # 最多保存的模型数量
#saver.restore(sess, best_model_path)



# -----------
# 创建新的变量，并单独对这部分进行初始化和训练
# 参考：https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
# 参考：https://stackoverflow.com/questions/35013080/tensorflow-how-to-get-all-variables-from-rnn-cell-basiclstm-rnn-cell-multirnn

def textcnn(cnn_inputs, n_step):
    """build the TextCNN network. Return the h_drop"""
    # cnn_inputs.shape = [batchsize, n_step, hidden_size*2+embedding_size] 
    inputs = tf.expand_dims(cnn_inputs, -1)
    pooled_outputs = list()
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, hidden_size*2+embedding_size, 1, n_filter]
            W_filter = weight_variable(shape=filter_shape, name="W_filter")
            beta = bias_variable(shape=[n_filter], name="beta")
            tf.summary.histogram('beta', beta)
            conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            conv_bn, update_ema = batchnorm(conv, tst, n_updates, beta, convolutional=True)    # 在激活层前面加 BN
            # Apply nonlinearity, batch norm scaling is not useful with relus
            # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
            h = tf.nn.relu(conv_bn, name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,ksize=[1, n_step - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],padding='VALID',name="pool")
            pooled_outputs.append(pooled)
            update_emas.append(update_ema)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, n_filter_total])
    return h_pool_flat    # shape = [batch_size, n_filter_total]

with tf.variable_scope('new_scope') as vs:
    fc_keep_prob = tf.placeholder(tf.float32, [])
    with tf.variable_scope('title_cnn'):
        title_inputs = tf.concat([word_encoder_title, inputs1], axis=2)
        output_title = textcnn(title_inputs, n_step1) # shape = [batch_size, n_filter_total]
    with tf.variable_scope('content_cnn'):
        content_inpus = tf.concat([word_encoder_content, inputs2], axis=2)
        output_content = textcnn(content_inpus, n_step2) # shape = [batch_size, n_filter_total]
    with tf.variable_scope('fc-bn-layer'):
        output = tf.concat([output_title, output_content], axis=1)
        W_fc = weight_variable(shape=[n_filter_total*2, fc_hidden_size], name='Weight_fc')
        tf.summary.histogram('W_fc', W_fc)
        h_fc = tf.matmul(output, W_fc, name='h_fc')
        beta_fc = bias_variable(shape=[fc_hidden_size], name="beta_fc")
        tf.summary.histogram('beta_fc', beta_fc)
        fc_bn, update_ema_fc = batchnorm(h_fc, tst, n_updates, beta_fc, convolutional=False)
        update_emas.append(update_ema_fc)
        fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
        fc_drop = tf.nn.dropout(fc_bn_relu, fc_keep_prob)
    with tf.variable_scope('out_layer'):
        W_out = weight_variable(shape=[fc_hidden_size, n_class], name='Weight_out') 
        tf.summary.histogram('Weight_out', W_out)
        b_out = bias_variable(shape=[n_class], name='bias_out') 
        tf.summary.histogram('bias_out', b_out)
        y_pred = tf.nn.xw_plus_b(fc_drop, W_out, b_out, name='y_pred')  #每个类别的分数 scores
    with tf.variable_scope('cost'):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_inputs))
        tf.summary.scalar('cost', cost)
    new_variables1 = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

# optimizers,先用 train_op2 训练，再用 train_op1  
with tf.variable_scope('optimizers') as vs:
    with tf.variable_scope('AdamOptimizer1'):
        tvars1 = tf.trainable_variables()
        grads1, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars1), max_grad_norm)
        optimizer1 = tf.train.AdamOptimizer(learning_rate=lr)
        train_op1 = optimizer1.apply_gradients(zip(grads1, tvars1),
            global_step=tf.contrib.framework.get_or_create_global_step())
        
    with tf.variable_scope('AdamOptimizer2'):          # 只更新新创建的部分变量
        tvars2 = new_variables1
        grads2 = tf.gradients(cost, tvars2)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=lr)
        train_op2 = optimizer2.apply_gradients(zip(grads2, tvars2),
            global_step=tf.contrib.framework.get_or_create_global_step())
    update_op = tf.group(*update_emas)   # 更新 BN 参数
    merged = tf.summary.merge_all() # summary
    train_writer = tf.summary.FileWriter(summary_path + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(summary_path + 'test')
    new_variables2 = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

print('Initialize new variables')
new_variables = new_variables1 + new_variables2
init_new = tf.variables_initializer(new_variables)
sess.run(init_new)
print('Finished creating the birnn-cnn model.')


#-------------------导入数据
import sys
sys.path.append('..')
from data_helpers import BatchGenerator
from data_helpers import to_categorical
from evaluator import score_eval

save_path = '../data/'
print('loading data...')
time0 = time.time()
X_title = np.load(save_path+'wd_train_title.npy') # X_tr_title_50
sample_num = X_title.shape[0]
X_content = np.load(save_path+'wd_train_content.npy') # X_tr_content_150
X = np.hstack([X_title, X_content])
y = np.load(save_path+'y_tr.npy')
print('finished loading data, time cost %g s' % (time.time() - time0))

# 划分验证集
np.random.seed(seed_num)
new_index = np.random.permutation(sample_num)
X = X[new_index]
y = y[new_index]
X_valid = X[:valid_num]
y_valid = y[:valid_num]
X_train = X[valid_num:]
y_train = y[valid_num:]
print('train_num=%d, valid_num=%d' % (X_train.shape[0], X_valid.shape[0]))

# 构建数据生成器
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
print('X_train.shape=', X_train.shape, 'y_train.shape=', y_train.shape)
print('X_valid.shape=', X_valid.shape, 'y_valid.shape=', y_valid.shape)

del X
del y
del X_train
del y_train

marked_labels_list = data_valid.y.tolist() # 所有的标注结果
valid_data_size = data_valid.y.shape[0]
def valid_epoch():
    """Testing or valid."""
    data_valid._index_in_epoch = 0  # 先指向第一个值
    _batch_size = te_batch_size
    fetches = [cost, y_pred]   
    batch_num = int(valid_data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    predict_labels_list = list()  # 所有的预测结果
    for i in range(batch_num):
        X_batch, y_batch = data_valid.next_batch(_batch_size)
        X1_batch = X_batch[:, :n_step1]
        X2_batch = X_batch[:, n_step1:]
        y_batch = to_categorical(y_batch)
        feed_dict = {X1_inputs:X1_batch, X2_inputs:X2_batch, y_inputs:y_batch, lr:1e-5, 
                    batch_size:_batch_size, keep_prob:1.0, fc_keep_prob:1.0,tst:True, n_updates:global_step}
        _cost, predict_labels = sess.run(fetches, feed_dict)
        _costs += _cost    
        predict_labels = map(lambda label: label.argsort()[-1:-6:-1], predict_labels) # 取最大的5个下标
        predict_labels_list.extend(predict_labels)
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)
    mean_cost = _costs / batch_num
    return mean_cost, precision, recall, f1

print('Finised loading data, time %g s' % (time.time() - time0))


#--------训练模型
_lr = 6e-4
decay = 0.85
max_epoch = 3            # cpu迭代次数
max_max_epoch = 6        # 最多迭代的次数
valid_step = 10000       # 每 valid_step 就进行一次 valid 运算
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
print('tr_batch_num=%d' % tr_batch_num)
last_f1 = 0.40
model_num = 0
global_step = 0


time0 = time.time()
train_op = train_op2
for epoch in range(max_max_epoch):
    if epoch == max_epoch:
        train_op = train_op1
        if model_num == 0:
            model_num += 1
            save_path = saver.save(sess, model_path, global_step=model_num)
            print('the save path is ', save_path)
        print('Begin updating embedding.')
    print('EPOCH %d, _lr= %g' % (epoch+1, _lr))
    for batch in range(tr_batch_num): 
        global_step += 1
        if (global_step+1) % valid_step == 0:    # 进行 valid 计算
            valid_cost, precision, recall, f1 = valid_epoch()
            _lr = _lr*decay
            print('Global_step=%d: valid cost=%g; p=%g, r=%g, f1=%g, time=%g s' % (
                    global_step, valid_cost, precision, recall, f1, time.time()-time0))
            time0 = time.time()
            if (f1 > last_f1):
                last_f1 = f1
                model_num += 1
                save_path = saver.save(sess, model_path, global_step=model_num)
                print('the save path is ', save_path) 
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        X1_batch = X_batch[:, :n_step1]
        X2_batch = X_batch[:, n_step1:]
        y_batch = to_categorical(y_batch)
        feed_dict = {X1_inputs:X1_batch, X2_inputs:X2_batch, y_inputs:y_batch, batch_size:tr_batch_size, lr:_lr,
                     keep_prob:0.5, fc_keep_prob:0.5, tst:False, n_updates:global_step}
        fetches = [merged, cost, train_op, update_op]
        summary, _cost, _, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
        if global_step % 100:
            train_writer.add_summary(summary, global_step)
            X_batch, y_batch = data_valid.next_batch(tr_batch_size)
            X1_batch = X_batch[:, :n_step1]
            X2_batch = X_batch[:, n_step1:]
            y_batch = to_categorical(y_batch)
            feed_dict = {X1_inputs:X1_batch, X2_inputs:X2_batch,  y_inputs:y_batch,lr:1e-4,
                         batch_size:tr_batch_size, keep_prob:1.0, fc_keep_prob:1.0, tst:True, n_updates:global_step}
            fetches = [merged, cost]
            summary, _cost = sess.run(fetches, feed_dict)
            test_writer.add_summary(summary, global_step)
        
        
valid_cost, precision, recall, f1 = valid_epoch()  # # 每个 epoch 进行一次验证 valid
print('END>Global_step=%d;  valid cost=%g; p=%g, r=%g, f1=%g; speed=%g s/epoch' % (
    global_step, valid_cost, precision, recall, f1, time.time()-time0) )
if (f1 > last_f1):
    model_num += 1
    save_path = saver.save(sess, model_path, global_step=model_num)
    print('the save path is ', save_path) 