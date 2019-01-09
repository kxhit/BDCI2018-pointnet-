import argparse
import math
from datetime import datetime
# import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)  # model
sys.path.append(ROOT_DIR)  # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util

sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))

#kx
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models import pointnet2_sem_seg_xyzi_final as MODEL

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg_xyzi_final', help='Model name [default: pointnet2_sem_seg_xyzi_final.py]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=3, help='Epoch to run [default: 3]')
parser.add_argument('--max_loop', type=int, default=3, help='Loop to run [default: 3]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--part_num', type=int, default=10, help='How many part is train_set divided')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
MAX_LOOP = FLAGS.max_loop
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


best_acc = 0
best_iou = 0

NUM_CLASSES = 8

#取数据
def get_data(files_list):
    data_batch_list = []
    label_batch_list = []
    for h5_filename in files_list:
        data_batch, label_batch = provider.loadDataFile(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    print(data_batches.shape)
    print(label_batches.shape)
    return data_batches, label_batches,label_batch_list

#取类别权重
def get_weights(label_batch_list):
    # get weights
    label_batches = np.concatenate(label_batch_list, 0)
    weights = np.zeros(NUM_CLASSES)
    for seg in label_batch_list:
        tmp, _ = np.histogram(seg, list(range(NUM_CLASSES + 1)))
        weights += tmp
    print(weights)
    weights = weights.astype(np.float32)
    weights = weights / np.sum(weights)
    np.savetxt(os.path.join(LOG_DIR, 'class_distribution.txt'), weights)  # 存样本分布概率
    # weights_batchs = 1 / (np.log(1.2 + label_weights))**2
    weights_array = np.abs(np.log(weights))

    print(weights_array)
    np.savetxt(os.path.join(LOG_DIR, 'weights.txt'), weights_array)

    weights_batchs = np.zeros(label_batches.shape)
    for index in range(NUM_CLASSES):
        weights_batchs[np.where(label_batches == index)] = weights_array[index]

    print(weights_batchs.shape)
    return weights_batchs

#建图
def create_graph(iter):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            # pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            "--- Get model and loss"
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=500)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        sess = tf.Session(config=config)
        if iter == 0:
            log_string("First model is ready!")
            init = tf.global_variables_initializer()
            sess.run(init, {is_training_pl: True})
        elif os.path.exists(os.path.join(LOG_DIR, "model{}.ckpt.meta".format(iter-1))):
            log_string("Model has been recovered last model, from file: %s" % os.path.join(LOG_DIR,
                                                                                           "model{}.ckpt".format(
                                                                                               iter - 1)))
            saver.restore(sess, os.path.join(LOG_DIR, "model{}.ckpt".format(iter-1)))

        else:
            log_string("Last model didn't exist, create new model")
            init = tf.global_variables_initializer()
            sess.run(init, {is_training_pl: True})

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train{}'.format(iter)),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # kx
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        return sess, ops, saver, train_writer, test_writer

#跑1个iter数据
def train_one_iteration(part_files, iter):
    train_data, train_label, _ = get_data(part_files)
    print("train_data shape:", train_data.shape, train_label.shape)

    distributions, _ = np.histogram(train_label.reshape(-1), list(range(NUM_CLASSES + 1)))
    print("train_data shape:", train_data.shape, train_label.shape)
    distributions = distributions.astype(np.float32)
    distributions = distributions / np.sum(distributions)
    np.savetxt(os.path.join(LOG_DIR, 'class_distribution{}.txt'.format(iter)), distributions)  # 存样本分布概率
    # sess, ops, saver, train_writer, test_writer = create_graph(iter)

    for epoch in tqdm(range(MAX_EPOCH)):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(sess, ops, train_writer, train_data, train_label)
        if(epoch % 5 ==0) or (epoch == MAX_EPOCH-1):
            acc, iou = eval_one_epoch(sess, ops, test_writer, test_data, test_label)
            # print("acc", acc)
            # print("iou", iou)
            global best_acc
            global best_iou
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_acc_iter{}_epoch_{}.ckpt".format(iter, epoch)))
                log_string("Model best acc saved in file: %s" % save_path)
                log_string("best_acc: {}".format(best_acc))
            if iou > best_iou:
                best_iou = iou
                save_path = saver.save(sess,
                                       os.path.join(LOG_DIR, "best_iou_iter{}_epoch_{}.ckpt".format(iter, epoch)))
                log_string("Model best iou saved in file: %s" % save_path)
                log_string("best_iou: {}".format(best_iou))

        # Save the variables to disk.
        if (epoch % 10 == 0) or (epoch == MAX_EPOCH-1):
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model{}.ckpt".format(iter)))
            log_string("Model saved in file: %s" % save_path)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw
    return batch_data, batch_label, batch_smpw


def train_one_epoch(sess, ops, train_writer,train_data, train_label):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    train_weights = np.ones(train_label.shape)

    current_data, current_label, _, current_weights = provider.shuffle_data_with_weights(train_data[:, 0:NUM_POINT, :],
                                                                                         train_label, train_weights)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, 0:4],#xyz i
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: current_weights[start_idx:end_idx],
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        # correct = np.sum(pred_val == batch_label)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        if (batch_idx + 1) % 100 == 0:#10
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0


# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer, test_data, test_label):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    # test_idxs = np.arange(0, len(TEST_DATASET))
    # num_batches = len(TEST_DATASET) // BATCH_SIZE #kx

    current_data = test_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(test_label)
    current_weights = np.ones(current_label.shape)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    # 应当有个最终的IOU 根据pred和gt的交集/并集的比值
    union = np.zeros(NUM_CLASSES)
    intersection = np.zeros(NUM_CLASSES)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
    print('num_batches ',num_batches)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, 0:4],#xyz i
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: current_weights[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)  # BxN
        # correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (
        #             batch_smpw > 0))  # evaluate only on 20 categories but not unknown
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        # total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        # tmp, _ = np.histogram(batch_label, list(range(22)))
        # labelweights += tmp

        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)

                if pred_val[i-start_idx, j] == l:
                    union[l] = union[l] + 1
                    intersection[l] = intersection[l] + 1
                else:
                    union[pred_val[i-start_idx, j]] = union[pred_val[i-start_idx, j]] + 1
                    union[l] = union[l] + 1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    for id in range(NUM_CLASSES):
        log_string('class %d correct %f / %f' % (id, total_correct_class[id], total_seen_class[id]))
        if total_seen_class[id] != 0:
            log_string('class %d accuracy: %f' % (id, total_correct_class[id] / total_seen_class[id]))

    ##IOU
    result_pred = np.zeros(NUM_CLASSES)
    num_pred = np.zeros(NUM_CLASSES)
    # class=0 是背景类 不做计算 算1-7共7类的IOU的均值
    for c in range(NUM_CLASSES - 1):
        cl = c + 1
        if (union[cl] == 0):
            continue
        else:
            result_pred[cl] = float(result_pred[cl]) + float(intersection[cl]) / float(union[cl])
            num_pred[cl] = num_pred[cl] + 1

    result = np.zeros(NUM_CLASSES + 1)  # 末尾是均值
    for c in range(NUM_CLASSES - 1):
        cl = c + 1
        result[cl] = 0.0 if (num_pred[cl] == 0) else result_pred[cl] / float(num_pred[cl])
        result[NUM_CLASSES] = result[cl] + result[NUM_CLASSES]

    result[0] = 0.0
    result[NUM_CLASSES] = result[NUM_CLASSES] / float(NUM_CLASSES - 1)

    for id in range(NUM_CLASSES - 1):
        log_string('class %d IOU: %f' % (id + 1, result[id + 1]))
    log_string('IOU eval: %f' % result[NUM_CLASSES])

    EPOCH_CNT += 1
    return total_correct / float(total_seen), result[NUM_CLASSES]


if __name__ == "__main__":
    # please change these paths below
    train_files_path = '/media/data/train/outdoor_sem_seg_hdf5_data_train_part_25000_30_9_drop0.001/all_files.txt'
    test_files_path = '/media/data/train/outdoor_sem_seg_hdf5_data_test_part_25000_30_9_drop0.001/all_files.txt'
    part_num = FLAGS.part_num
    train_files = provider.getDataFiles(train_files_path)  # train set
    test_files = provider.getDataFiles(test_files_path)
    test_data, test_label, _ = get_data(test_files)
    np.random.seed(0)
    test_idx = np.random.permutation(np.arange(test_data.shape[0]))
    test_data = test_data[test_idx[0:BATCH_SIZE*500],:,:]
    test_label = test_label[test_idx[0:BATCH_SIZE * 500], :]
    print("test_data shape:", test_data.shape, test_label.shape)

    # 0 if you want to train from beginning, n if you want to restore from model n (1, 2 or ...)
    sess, ops, saver, train_writer, test_writer = create_graph(0)

    for loop in tqdm(range(MAX_LOOP)):# run total samples MAX_LOOP times
        np.random.shuffle(train_files)
        log_string('==========================================================')
        log_string("loop: %d" % loop)
        for i in tqdm(range(part_num)):  # the training data is divided into part_num parts
            log_string('------------------------------------------------------------')
            log_string('part: %d' % i)
            part_files = list(np.array(train_files)[i::part_num])
            train_one_iteration(part_files, i)  # train one part for max_epoch times
    LOG_FOUT.close()
    pass
