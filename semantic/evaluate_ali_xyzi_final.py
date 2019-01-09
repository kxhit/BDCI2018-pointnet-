import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)  # model
sys.path.append(ROOT_DIR)  # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# from model import *
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models import pointnet2_sem_seg_xyzi_final as MODEL
import indoor3d_util_test as indoor3d_util
import numpy as np
import math
# from insert_file import *
import time
from tqdm import tqdm
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point number [default: 1024]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)]

NUM_CLASSES = 8

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):#gpu
        pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(None, NUM_POINT)#BATCH_SIZE
        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, num_class=NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
                  'labels_pl': labels_pl,
                  'smpws_pl': smpws_pl,
                  'is_training_pl': is_training_pl,
                  'pred': pred,
                  'pred_softmax': pred_softmax,
                  'loss': loss}

    total_correct = 0
    total_seen = 0
    total_IOU = 0
    fout_out_filelist = open(FLAGS.output_filelist, 'w')
    for room_path in tqdm(ROOM_PATH_LIST):
        # out_label_filename = os.path.basename(room_path)[:-4] + '.csv'
        out_data_label_filename = os.path.basename(room_path)[:-4] + '_pred.txt'
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        out_gt_label_filename = os.path.basename(room_path)[:-4] + '_gt.txt'
        out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
        print((room_path, out_data_label_filename))
        time_start = time.time()
        a, b, c= eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename)
        time_end = time.time()
        print('time cost: ',time_end-time_start)
        log_string('time cost: %f' % (time_end - time_start))
        total_correct += a
        total_seen += b
        total_IOU += c
        fout_out_filelist.write(out_data_label_filename+'\n')
    fout_out_filelist.close()
    log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('all IOU eval: %f' % (float(total_IOU)/float(len(ROOM_PATH_LIST))))

def eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    if FLAGS.visu:
        fout = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_pred.obj'), 'w')
        # fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_gt.obj'), 'w')
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')
    pred_label_csv_filename = os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4] + '.csv')
    fout_pred_label_csv = open(pred_label_csv_filename, 'w')

    time_sample_start = time.time()
    current_data, current_label, current_id = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT,block_size=5.0, stride=5.0,drop=1.0) #XYZRGBX'Y'Z' L
    # current_data, current_label = indoor3d_util.room2blocks_wrapper(room_path, NUM_POINT, block_size=5.0,
    #                                                                            stride=3.0)  # XYZRGB L

    time_sample_end = time.time()
    print("sample data cost time: ", time_sample_end - time_sample_start)
    log_string("sample data cost time: %f" % (time_sample_end - time_sample_start))

    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)
    current_id = np.squeeze(current_id)
    print("current shape")
    print(current_data.shape)# 618x1024x9
    print(current_label.shape)# 618x1024
    print(current_id.shape)#618x1024
    # print(current_data[0:3,0,:])
    current_id_len = len(np.unique(current_id))
    print(current_id_len)

    # kx
    current_weights = np.ones(current_label.shape)

    # Get room dimension..
    #kx
    file_type = os.path.splitext(room_path)[-1]
    if file_type == '.txt':
        data_label = np.loadtxt(room_path)
    elif file_type == '.npy':
        data_label = np.load(room_path)
    else: print("file type wrong!")

    assert current_id_len == data_label.shape[0]

    #初始化文件 以便于后面写入指定行
    for line_i in range(data_label.shape[0]):
        fout_pred_label_csv.write("\n")
        fout_data_label.write("\n")
        fout_gt_label.write("\n")
    fout_pred_label_csv.close()
    fout_data_label.close()
    fout_gt_label.close()

    f = open(pred_label_csv_filename, 'r')
    fout_pred_label_csv_list = f.readlines()
    f.close()
    f = open(out_data_label_filename, 'r')
    fout_data_label_list = f.readlines()
    f.close()
    f = open(out_gt_label_filename, 'r')
    fout_gt_label_list = f.readlines()
    f.close()

    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')
    fout_pred_label_csv = open(pred_label_csv_filename, 'w')

    #kx
    # data_label = np.load(room_path)
    data = data_label[:,0:6]
    #kx
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    #kx
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE # // 整除取整
    tail_batch_num = file_size-num_batches*BATCH_SIZE
    print(file_size)

    index_list = []
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, 0:4],#xyz i
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: current_weights[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
        # print("pred_val shape: %d",pred_val.shape[2])
        if FLAGS.no_clutter:
            # pred_label = np.argmax(pred_val[:,:,0:NUM_CLASSES-1], 2) # BxN
            pred_label = np.argmax(pred_val[:, :, 0:NUM_CLASSES], 2)  # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN
        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx+b, :, :]
            l = current_label[start_idx+b,:]
            index = current_id[start_idx+b,:]
            unique_index, unique_index_id = np.unique(index,return_index=True)
            pts = current_data[start_idx + b, unique_index_id, :]
            l = current_label[start_idx + b, unique_index_id]
            index = current_id[start_idx + b, unique_index_id]

            #kx
            pts[:,6] *= max_room_x
            pts[:,7] *= max_room_y
            pts[:,8] *= max_room_z
            #kx
            pts[:,3:6] *= 255.0
            pred = pred_label[b, :]
            exsited_probability = 1.0
            for i in tqdm(range(len(unique_index_id))):#NUM_POINT
                color = indoor3d_util.g_label2color[pred[i]]
                # color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
                if FLAGS.visu:
                    fout.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color[0], color[1], color[2]))
                    # fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_gt[0], color_gt[1], color_gt[2]))

                    # fout.write('v %f %f %f %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], color[0], color[1], color[2]))
                    # fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], color_gt[0], color_gt[1], color_gt[2]))

                # 不是最后一行
                if ((fout_pred_label_csv_list[int(index[i])] != '\n') and (index[i] != data_label.shape[0]))\
                        or ((fout_pred_label_csv_list[int(index[i])] != '') and (index[i] == data_label.shape[0])):#是最后一行
                #已经有值 末尾为 其他行label \n / 最后一行 label(无\n)
                    # if (index[i] != data_label.shape[0]):#不是最后一行
                    #     exsited_label = (fout_pred_label_csv_list[int(index[i])])[-2] #已预测的label
                    #     exsited_probability = (fout_data_label_list[int(index[i])])[-4]  #已预测的概率
                    # else:#最后一行
                    #     exsited_label = (fout_pred_label_csv_list[int(index[i])])[-1] #已预测的label
                    #     exsited_probability = (fout_data_label_list[int(index[i])])[-3]  #已预测的概率
                    exsited_label = (fout_pred_label_csv_list[int(index[i])])[0]  # 已预测的label
                    if pred_val[b, i, pred[i]] > float(exsited_probability):#当前概率高于之前的已预测值
                        if pred[i] != int(exsited_label):
                            print("pred is changed from %d to %d" % (int(exsited_label),pred[i]))
                        #直接赋值
                        if (index[i] == data_label.shape[0]):  # 最后一行不用换行了 '\n'
                            fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(
                                pts[i, 8]) + ' ' + str(pts[i, 3]) + \
                                                                  ' ' + str(pts[i, 4]) + ' ' + str(
                                pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(pred[i])
                            fout_gt_label_list[int(index[i])] = str(l[i])
                            fout_pred_label_csv_list[int(index[i])] = str(pred[i])
                        else:
                            fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(
                                pts[i, 8]) + \
                                                                  ' ' + str(pts[i, 3]) + ' ' + str(
                                pts[i, 4]) + ' ' + str(pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(
                                pred[i]) + '\n'
                            fout_gt_label_list[int(index[i])] = str(l[i]) + '\n'
                            fout_pred_label_csv_list[int(index[i])] = str(pred[i]) + '\n'
                    else:#没有之前高 跳过
                        continue
                else:#当前还没有值
                    # 直接赋值
                    if(index[i]==data_label.shape[0]):#最后一行不用换行了 '\n'
                        fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(pts[i, 8]) +' ' + str(pts[i, 3]) + \
                                                              ' ' + str(pts[i, 4]) + ' ' + str(pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(pred[i])
                        fout_gt_label_list[int(index[i])] = str(l[i])
                        fout_pred_label_csv_list[int(index[i])] = str(pred[i])
                    else:
                        fout_data_label_list[int(index[i])] = str(pts[i,6])+' '+str(pts[i,7])+' '+str(pts[i,8])+\
                                                              ' '+str(pts[i,3])+' '+str(pts[i,4])+' '+str(pts[i,5])+' '+str(pred_val[b,i,pred[i]])+' '+str(pred[i])+'\n'
                        fout_gt_label_list[int(index[i])] = str(l[i])+'\n'
                        fout_pred_label_csv_list[int(index[i])] = str(pred[i])+'\n'

                exsited_probability = pred_val[b, i, pred[i]]

        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)

        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                if l >= 7: l=7 # assert l==8
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    if tail_batch_num > 0:
        batch_idx = num_batches
        start_idx = batch_idx * BATCH_SIZE
        end_idx = batch_idx * BATCH_SIZE + tail_batch_num
        cur_batch_size = end_idx - start_idx

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, 0:4],  # xyz i
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: current_weights[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
        # print("pred_val shape: %d",pred_val.shape[2])
        if FLAGS.no_clutter:
            # pred_label = np.argmax(pred_val[:,:,0:NUM_CLASSES-1], 2) # BxN
            pred_label = np.argmax(pred_val[:, :, 0:NUM_CLASSES], 2)  # BxN
        else:
            pred_label = np.argmax(pred_val, 2)  # BxN
        # Save prediction labels to OBJ file
        for b in range(tail_batch_num):
            pts = current_data[start_idx + b, :, :]
            l = current_label[start_idx + b, :]
            index = current_id[start_idx + b, :]



            # kx
            pts[:, 6] *= max_room_x
            pts[:, 7] *= max_room_y
            pts[:, 8] *= max_room_z
            # kx
            pts[:, 3:6] *= 255.0
            pred = pred_label[b, :]
            exsited_probability = 1.0
            for i in tqdm(range(NUM_POINT)):
                color = indoor3d_util.g_label2color[pred[i]]
                # color_gt = indoor3d_util.g_label2color[current_label[start_idx + b, i]]
                if FLAGS.visu:
                    fout.write(
                        'v %f %f %f %d %d %d\n' % (pts[i, 6], pts[i, 7], pts[i, 8], color[0], color[1], color[2]))
                    # fout_gt.write('v %f %f %f %d %d %d\n' % (
                    # pts[i, 6], pts[i, 7], pts[i, 8], color_gt[0], color_gt[1], color_gt[2]))
                    # fout.write('v %f %f %f %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], color[0], color[1], color[2]))
                    # fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], color_gt[0], color_gt[1], color_gt[2]))

                # 不是最后一行
                if ((fout_pred_label_csv_list[int(index[i])] != '\n') and (index[i] != data_label.shape[0])) \
                        or ((fout_pred_label_csv_list[int(index[i])] != '') and (
                        index[i] == data_label.shape[0])):  # 是最后一行
                    # 已经有值 末尾为 其他行label \n / 最后一行 label(无\n)
                    # if (index[i] != data_label.shape[0]):#不是最后一行
                    #     exsited_label = (fout_pred_label_csv_list[int(index[i])])[-2] #已预测的label
                    #     exsited_probability = (fout_data_label_list[int(index[i])])[-4]  #已预测的概率
                    # else:#最后一行
                    #     exsited_label = (fout_pred_label_csv_list[int(index[i])])[-1] #已预测的label
                    #     exsited_probability = (fout_data_label_list[int(index[i])])[-3]  #已预测的概率
                    exsited_label = (fout_pred_label_csv_list[int(index[i])])[0]  # 已预测的label
                    if pred_val[b, i, pred[i]] > float(exsited_probability):  # 当前概率高于之前的已预测值
                        if pred[i] != int(exsited_label):
                            print("pred is changed from %d to %d" % (int(exsited_label), pred[i]))
                        # 直接赋值
                        if (index[i] == data_label.shape[0]):  # 最后一行不用换行了 '\n'
                            fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(
                                pts[i, 8]) + ' ' + str(pts[i, 3]) + \
                                                                  ' ' + str(pts[i, 4]) + ' ' + str(
                                pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(pred[i])
                            fout_gt_label_list[int(index[i])] = str(l[i])
                            fout_pred_label_csv_list[int(index[i])] = str(pred[i])
                        else:
                            fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(
                                pts[i, 8]) + \
                                                                  ' ' + str(pts[i, 3]) + ' ' + str(
                                pts[i, 4]) + ' ' + str(pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(
                                pred[i]) + '\n'
                            fout_gt_label_list[int(index[i])] = str(l[i]) + '\n'
                            fout_pred_label_csv_list[int(index[i])] = str(pred[i]) + '\n'
                    else:  # 没有之前高 跳过
                        continue
                else:  # 当前还没有值
                    # 直接赋值
                    if (index[i] == data_label.shape[0]):  # 最后一行不用换行了 '\n'
                        fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(
                            pts[i, 8]) + ' ' + str(pts[i, 3]) + \
                                                              ' ' + str(pts[i, 4]) + ' ' + str(
                            pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(pred[i])
                        fout_gt_label_list[int(index[i])] = str(l[i])
                        fout_pred_label_csv_list[int(index[i])] = str(pred[i])
                    else:
                        fout_data_label_list[int(index[i])] = str(pts[i, 6]) + ' ' + str(pts[i, 7]) + ' ' + str(
                            pts[i, 8]) + \
                                                              ' ' + str(pts[i, 3]) + ' ' + str(
                            pts[i, 4]) + ' ' + str(pts[i, 5]) + ' ' + str(pred_val[b, i, pred[i]]) + ' ' + str(
                            pred[i]) + '\n'
                        fout_gt_label_list[int(index[i])] = str(l[i]) + '\n'
                        fout_pred_label_csv_list[int(index[i])] = str(pred[i]) + '\n'

                exsited_probability = pred_val[b, i, pred[i]]

            correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(start_idx, end_idx):
                for j in range(NUM_POINT):
                    l = current_label[i, j]
                    if l >= NUM_CLASSES-1: l=NUM_CLASSES-1 #assert l==7
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_label[i - start_idx, j] == l)

    fout_data_label.writelines(fout_data_label_list)
    fout_gt_label.writelines(fout_gt_label_list)
    fout_pred_label_csv.writelines(fout_pred_label_csv_list)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    for id in range(NUM_CLASSES):
        log_string('class %d correct %f / %f' % (id, total_correct_class[id], total_seen_class[id]))
        if total_seen_class[id] != 0:
            log_string('class %d accuracy: %f'% (id, total_correct_class[id]/total_seen_class[id]))
    #应当有个最终的IOU 根据pred和gt的交集/并集的比值
    union = np.zeros(NUM_CLASSES)
    intersection = np.zeros(NUM_CLASSES)
    for i in range(data_label.shape[0]):
        pred = int(fout_pred_label_csv_list[i][0])
        gt = int(fout_gt_label_list[i][0])
        if gt >=7: gt=7 # assert gt == 7
        if pred == gt:
            union[pred] = union[pred] + 1
            intersection[pred] = intersection[pred] + 1
        else:
            union[pred] = union[pred] + 1
            union[gt] = union[gt] + 1
    result_pred = np.zeros(NUM_CLASSES)
    num_pred = np.zeros(NUM_CLASSES)
    #class=0 是背景类 不做计算 算1-7共7类的IOU的均值
    for c in range(NUM_CLASSES-1):
        cl = c+1
        if(union[cl]==0):
            continue
        else:
            result_pred[cl] = float(result_pred[cl]) + float(intersection[cl])/float(union[cl])
            num_pred[cl] = num_pred[cl] + 1

    result = np.zeros(NUM_CLASSES+1) #末尾是均值
    for c in range(NUM_CLASSES-1):
        cl = c+1
        result[cl] = 0.0 if (num_pred[cl]==0) else result_pred[cl] / float(num_pred[cl])
        # result[cl] = (num_pred[cl] == 0)? 0.0f: result_pred[cl] / (float)num_pred[cl]
        result[NUM_CLASSES] = result[cl] + result[NUM_CLASSES]

    result[0] = 0.0
    result[NUM_CLASSES] = result[NUM_CLASSES] / float(NUM_CLASSES-1)

    for id in range(NUM_CLASSES-1):
        log_string('class %d IOU: %f' % (id+1, result[id+1]))
    log_string('IOU eval: %f' % result[NUM_CLASSES])

    # #IoU with background
    # # class=0 是背景类 做计算 算0-7共8类的IOU的均值
    # result_pred = np.zeros(NUM_CLASSES)
    # num_pred = np.zeros(NUM_CLASSES)
    # for c in range(NUM_CLASSES):
    #     cl = c
    #     if (union[cl] == 0):
    #         continue
    #     else:
    #         result_pred[cl] = float(result_pred[cl]) + float(intersection[cl]) / float(union[cl])
    #         num_pred[cl] = num_pred[cl] + 1
    #
    # result = np.zeros(NUM_CLASSES+1)  # 末尾是均值
    # for c in range(NUM_CLASSES):
    #     cl = c
    #     result[cl] = 0.0 if (num_pred[cl] == 0) else result_pred[cl] / float(num_pred[cl])
    #     # result[cl] = (num_pred[cl] == 0)? 0.0f: result_pred[cl] / (float)num_pred[cl]
    #     result[NUM_CLASSES] = result[cl] + result[NUM_CLASSES]
    #
    # # result[0] = 0.0
    # result[NUM_CLASSES] = result[NUM_CLASSES] / float(NUM_CLASSES)
    #
    # for id in range(NUM_CLASSES):
    #     log_string('class %d IOU: %f' % (id, result[id]))
    # log_string('8 IOU eval: %f' % result[NUM_CLASSES])

    # #tensorflow mean_IoU
    # pred = [int(p[0]) for p in fout_pred_label_csv_list]
    # gt = [int(g[0]) for g in fout_gt_label_list]
    #
    # import cal_iou as zbq
    # weight = np.ones(len(gt))
    #
    # tf_IoU = zbq.cal_iou(gt,pred)#gt pred
    #
    # log_string('tensorflow mean 8 IoU: %f' % tf_IoU)
    #
    # weight[np.where(gt == 0)] = 0
    # tf_IoU = zbq.cal_iou(gt, pred)  # gt pred
    #
    # log_string('tensorflow mean 7 IoU: %f' % tf_IoU)

    fout_data_label.close()
    fout_gt_label.close()
    fout_pred_label_csv.close()
    if FLAGS.visu:
        fout.close()
        # fout_gt.close()
    return total_correct, total_seen, result[NUM_CLASSES]


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
