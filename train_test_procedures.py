# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import numpy as np
import random as rnd
from networks import network_model
from tensorflow.python.client import device_lib
import sys
import time
import shutil
import pandas as pd

import multiprocessing as multi_thread
from sklearn.metrics import roc_auc_score
from ops import auc




sys.path.append('..')
eps = 1e-9
initializer = 'XavierInit'
saver_mirrors_num = 3
lr_decay_factor = 0.999

print_separater="#########################################################"

def finding_continuous_dimensions(features):
    dimensions = features.shape[1]
    continous_dimension_idx = list()
    for ii in range(dimensions):

        current_dimension = np.unique(features[:,ii])
        discrete_dim=list()
        for jj in range(min(5,len(current_dimension))):
            curt_dim = current_dimension[jj]
            ceil_value = np.ceil(curt_dim)
            floor_value = np.floor(curt_dim)
            if ceil_value==floor_value:
                discrete_dim.append(True)
            else:
                discrete_dim.append(False)
        continuous_count = len([jj for jj in discrete_dim if jj == False])
        if continuous_count>0:
            continous_dimension_idx.append(ii)
    return continous_dimension_idx


def feature_omit(features, omit_idx):
    data_num = features.shape[0]
    keep_feature_count = 0
    for ii in range(features.shape[1]):
        if ii not in omit_idx:
            if keep_feature_count==0:
                output_feature = np.reshape(features[:,ii],[data_num,1])
            else:
                output_feature = np.concatenate([output_feature,np.reshape(features[:,ii],[data_num,1])], axis=1)
            keep_feature_count+=1
    return output_feature

def create_dataset(data_path, batch_size, shuffle):

    dataset = tf.contrib.data.make_csv_dataset(file_pattern=data_path,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_parallel_reads=multi_thread.cpu_count())
    dataset_iterator = dataset.make_initializable_iterator()
    get_next_dict = dataset_iterator.get_next()
    dict_keys = get_next_dict.keys()
    # dict_keys.sort()
    key_length = len(dict_keys)
    for ii in range(key_length-1):
        current_tensor = get_next_dict[dict_keys[ii+1]]
        current_tensor = tf.reshape(current_tensor,shape=[batch_size,1])
        current_tensor = tf.cast(current_tensor,tf.float32)
        if ii==0:
            output_data_tensor = current_tensor
        elif not ii == key_length-2:
            output_data_tensor = tf.concat([output_data_tensor, current_tensor], axis=1)
        elif ii==key_length-2:
            output_label_tensor = current_tensor

    return output_data_tensor, output_label_tensor, dataset_iterator

def r2_loss_calculas(labels, predictions, dimension_count):
    mean_true = tf.reduce_mean(labels)
    ssr = tf.reduce_sum(tf.square(predictions-mean_true)) / (int(labels.shape[0])*2-dimension_count-1)
    sst = tf.reduce_sum(tf.square(labels-mean_true)) / (int(labels.shape[0])-1) + eps
    r2_loss = ssr/sst

    return r2_loss


def train_procedures(args):
    previous_highest_auc = -1
    previous_highest_accuracy = -1
    previous_highest_r2 = -100
    previous_lowest_mse = 10000
    previous_highest_performance_at_epoch = -1
    print_test_info=''

    layer_channel_list_str = args.layer_channel_list.split(',')
    layer_channel_num_list = list()
    if not (len(layer_channel_list_str)==1 and layer_channel_list_str[0]==''):
        for ii in layer_channel_list_str:
            layer_channel_num_list.append(int(ii))
    layer_channel_num_list.append(1)

    dim_info = 'OmitNothing'


    local_device_protos = device_lib.list_local_devices()
    cpu_device = [x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if gpu_device == []:
        current_device = cpu_device[0]
    else:
        current_device = gpu_device[0]

    train_data_length = len(pd.read_csv(args.train_data_path))
    test_data_length = len(pd.read_csv(args.test_data_path))
    iteration_for_each_epoch_train = train_data_length / args.batch_size + 1
    iteration_for_each_epoch_test = test_data_length / args.batch_size + 1


    # real_feature_dimension = train_features.shape[1]
    # continous_dimension_idx = finding_continuous_dimensions(features=train_features)
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=config)

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(current_device):


                # get data tensor
                train_data_tensor, train_true_label_tensor, train_dataset_iterator \
                    = create_dataset(data_path=args.train_data_path,
                                     batch_size=args.batch_size,
                                     shuffle=True)
                test_data_tensor, test_true_label_tensor, test_dataset_iterator \
                    = create_dataset(data_path=args.test_data_path,
                                     batch_size=args.batch_size,
                                     shuffle=False)

                if args.training_mode==2 and not args.omit_dimension_idx=='':
                    omit_dimension = int(args.omit_dimension_idx)
                    if omit_dimension == 0:
                        add_tmp_tensor = tf.zeros(shape=[int(train_data_tensor.shape[0]),1],dtype=train_data_tensor.dtype)
                        train_data_tensor = tf.concat([add_tmp_tensor,train_data_tensor[:,1:]],axis=1)
                        test_data_tensor = tf.concat([add_tmp_tensor, test_data_tensor[:, 1:]], axis=1)
                    elif omit_dimension==19:
                        tensor_part_to_be_replaced = train_data_tensor[:,19:]
                        add_tmp_tensor = tf.zeros(shape=[int(train_data_tensor.shape[0]), int(tensor_part_to_be_replaced.shape[1])],
                                                  dtype=train_data_tensor.dtype)
                        train_data_tensor = tf.concat([train_data_tensor[:, 0:19], add_tmp_tensor], axis=1)
                        test_data_tensor = tf.concat([test_data_tensor[:, 0:19], add_tmp_tensor], axis=1)
                    else:
                        add_tmp_tensor = tf.zeros(shape=[int(train_data_tensor.shape[0]), 1],
                                                  dtype=train_data_tensor.dtype)
                        add_left = train_data_tensor[:,0:omit_dimension]
                        add_right = train_data_tensor[:,omit_dimension+1:]
                        train_data_tensor = tf.concat([add_left,add_tmp_tensor,add_right],axis=1)

                        add_left = test_data_tensor[:, 0:omit_dimension]
                        add_right = test_data_tensor[:, omit_dimension + 1:]
                        test_data_tensor = tf.concat([add_left, add_tmp_tensor, add_right], axis=1)
                else:
                    omit_dimension=-1


                # training framework vars
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                              dtype=tf.int64)
                epoch_step = tf.get_variable('epoch_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                             dtype=tf.int64)
                epoch_step_increase_one_op = tf.assign(epoch_step, epoch_step + 1)
                framework_vars = list()
                framework_vars.append(global_step)
                framework_vars.append(epoch_step)

                # learning rate init
                init_lr = args.init_lr
                learning_rate = tf.train.exponential_decay(learning_rate=init_lr,
                                                           global_step=global_step,
                                                           decay_steps=iteration_for_each_epoch_train,
                                                           decay_rate=lr_decay_factor,
                                                           staircase=True)
                learning_rate_summary = tf.summary.scalar('LearningRate', learning_rate)
                noise_added = tf.random_normal(shape=[args.batch_size, 1],
                                               mean=0.0,
                                               stddev=0.1,
                                               dtype=tf.float32)

                train_logits, train_probability, network_info = \
                    network_model(features=train_data_tensor + noise_added,
                                  device=current_device,
                                  is_training=True,
                                  weight_decay=True,
                                  weight_decay_rate=args.weight_decay_rate,
                                  initializer=initializer,
                                  layer_channel_num_list=layer_channel_num_list)

                train_probability_eval = \
                    network_model(features=train_data_tensor,
                                  device=current_device,
                                  is_training=False,
                                  weight_decay=False,
                                  weight_decay_rate=args.weight_decay_rate,
                                  initializer=initializer,
                                  reuse=True,
                                  layer_channel_num_list=layer_channel_num_list)[1]

                test_probability_eval = \
                    network_model(features=test_data_tensor,
                                  device=current_device,
                                  is_training=False,
                                  weight_decay=False,
                                  weight_decay_rate=args.weight_decay_rate,
                                  initializer=initializer,
                                  reuse=True,
                                  layer_channel_num_list=layer_channel_num_list)[1]

                # build losses
                # cross entropy losses for classification
                ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=train_true_label_tensor,
                                                                  logits=train_logits)
                ce_loss = tf.reduce_mean(ce_loss)
                ce_loss_summary = tf.summary.scalar("Loss_CE", ce_loss)

                auc_loss = auc(predictions=train_probability,
                               labels=train_true_label_tensor)
                auc_loss = 1 - auc_loss
                auc_loss = auc_loss * args.auc_loss_penalty
                auc_loss_summary = tf.summary.scalar("Loss_Auc", auc_loss/args.auc_loss_penalty)

                r2_loss = r2_loss_calculas(labels=train_true_label_tensor,
                                           predictions=train_probability,
                                           dimension_count=args.dimension_count)
                r2_loss = r2_loss * args.r2_loss_penalty
                r2_loss_summary = tf.summary.scalar("Loss_R2", r2_loss / args.r2_loss_penalty)


                # regularization with weight decay
                weight_decay_loss_list = tf.get_collection(network_info + '_weight_decay')
                weight_decay_loss = 0
                for ii in weight_decay_loss_list:
                    weight_decay_loss += ii
                wd_loss = weight_decay_loss / len(weight_decay_loss_list)
                wd_loss_summary = tf.summary.scalar("Loss_WD",
                                                    tf.abs(weight_decay_loss))

                loss_optimization = ce_loss + wd_loss + auc_loss + r2_loss
                merged_loss_summary = tf.summary.merge([ce_loss_summary, wd_loss_summary, auc_loss_summary, r2_loss_summary])
                merged_loss_value = [ce_loss, wd_loss, auc_loss]

                # build accuracies and errors
                test_prdt_label_tensor = (tf.sign(test_probability_eval - 0.5) + 1) / 2
                test_correct_prdt_tensor = tf.equal(test_prdt_label_tensor, test_true_label_tensor)
                test_classification_accuracy = tf.reduce_mean(tf.cast(test_correct_prdt_tensor, tf.float32)) * 100
                test_accuracy_summary = tf.summary.scalar('Accuracy_Test', test_classification_accuracy)

                train_prdt_label_tensor = (tf.sign(train_probability_eval - 0.5) + 1) / 2
                train_correct_prdt_tensor = tf.equal(train_prdt_label_tensor, train_true_label_tensor)
                train_classification_accuracy = tf.reduce_mean(tf.cast(train_correct_prdt_tensor, tf.float32)) * 100
                train_accuracy_summary = tf.summary.scalar('Accuracy_Train', train_classification_accuracy)


                test_mse_tensor = tf.square(test_probability_eval - test_true_label_tensor)
                test_mse_tensor = tf.reduce_mean(test_mse_tensor)
                test_error_summary = tf.summary.scalar('Error_Test', test_mse_tensor)

                train_mse_tensor = tf.square(train_probability_eval - train_true_label_tensor)
                train_mse_tensor = tf.reduce_mean(train_mse_tensor)
                train_error_summary = tf.summary.scalar('Error_Train', train_mse_tensor)

                merged_train_performance_summary = tf.summary.merge([train_accuracy_summary, train_error_summary])
                merged_test_performance_summary = tf.summary.merge([test_accuracy_summary, test_error_summary])
                merged_train_performance_value = [train_classification_accuracy, train_mse_tensor]
                merged_test_performance_value = [test_classification_accuracy, test_mse_tensor]

                # relevant variables
                trainable_vars = tf.trainable_variables()
                saving_vars = find_bn_avg_var(trainable_vars)

                # saver definition
                famework_saver = tf.train.Saver(max_to_keep=saver_mirrors_num, var_list=framework_vars)
                saver_full_model = tf.train.Saver(max_to_keep=saver_mirrors_num, var_list=saving_vars)

                # vars initialization
                exp_name = args.exp_name + '_' + network_info + '_' + dim_info
                current_model_save_path = os.path.join(args.model_save_path, exp_name)
                current_log_path = os.path.join(args.log_path, exp_name)

        sess.run(train_dataset_iterator.initializer)
        sess.run(test_dataset_iterator.initializer)
        if args.training_mode == 0:  # training from stratch
            tf.variables_initializer(var_list=saving_vars).run(session=sess)
            tf.variables_initializer(var_list=framework_vars).run(session=sess)
            if os.path.exists(current_model_save_path):
                shutil.rmtree(current_model_save_path)
            os.makedirs(current_model_save_path)
            os.makedirs(os.path.join(current_model_save_path, 'Model'))
            os.makedirs(os.path.join(current_model_save_path, 'Framework'))
            if os.path.exists(current_log_path):
                shutil.rmtree(current_log_path)
            os.makedirs(current_log_path)
        else:
            model_path = os.path.join(current_model_save_path, 'Model')
            framework_path = os.path.join(current_model_save_path, 'Framework')
            ckpt_model = tf.train.get_checkpoint_state(model_path)
            ckpt_framework = tf.train.get_checkpoint_state(framework_path)
            saver_full_model.restore(sess=sess, save_path=ckpt_model.model_checkpoint_path)
            famework_saver.restore(sess=sess, save_path=ckpt_framework.model_checkpoint_path)
            print('Full model restored from %s' % current_model_save_path)


        # summaries
        summary_writer = tf.summary.FileWriter(logdir=current_log_path,
                                               graph=sess.graph)

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss_optimization,
                                                                              var_list=trainable_vars,
                                                                              global_step=global_step)

        # info print for the complete of the initialization
        print("TrainSampleNum:%d,TestSampleNum:%d" %
              (train_data_length, test_data_length))
        print("BatchSize:%d, ItrsNum:%dX%d=%d, EpochNum:%d, DataReadThreads:%d" %
              (args.batch_size,
               iteration_for_each_epoch_train,
               args.epochs,
               iteration_for_each_epoch_train * args.epochs,
               args.epochs,
               multi_thread.cpu_count()))

        ei_start = epoch_step.eval(session=sess)
        global_step_start = global_step.eval(session=sess)
        ei_ranges = range(ei_start, args.epochs, 1)
        lr_start = learning_rate.eval(session=sess)
        print("Epoch:%d, GlobalStep:%d, LearningRate:%.10f" % (ei_start, global_step_start, lr_start))

        print("Initialization completed.")
        print(print_separater)
        print(print_separater)
        print(print_separater)
        print("AucLossPenalty:%.5f" % args.auc_loss_penalty)
        print("R2LossPenalty:%.5f" % args.r2_loss_penalty)
        raw_input("Press Enter to Coninue")
        print(print_separater)


        if args.training_mode==2:

            # tmp_train = sess.run(train_data_tensor)
            # tmp_test = sess.run(test_data_tensor)
            # evaluate
            test_accuracy, test_mse, test_auc, test_r2 = \
                evaluation(batch_size=args.batch_size,
                           dimension_count=args.dimension_count,

                           sess=sess,
                           logit_op=test_probability_eval, test_probability_op=test_probability_eval,
                           true_label_op=test_true_label_tensor,
                           prdt_op=test_prdt_label_tensor, true_op=test_true_label_tensor, mse_op=test_mse_tensor,
                           cort_op=test_correct_prdt_tensor,
                           actual_length=train_data_length,
                           iteration_for_each_epoch=iteration_for_each_epoch_test)
            train_accuracy, train_mse, train_auc, train_r2 = \
                evaluation(batch_size=args.batch_size,
                           dimension_count=args.dimension_count,
                           sess=sess,
                           logit_op=train_probability_eval, test_probability_op=train_probability_eval,
                           true_label_op=train_true_label_tensor,
                           prdt_op=train_prdt_label_tensor, true_op=train_true_label_tensor, mse_op=train_mse_tensor,
                           cort_op=train_correct_prdt_tensor,
                           actual_length=train_data_length,
                           iteration_for_each_epoch=iteration_for_each_epoch_train)

            print("Omit:%d, TrainAuc:%.5f, TrainAcy:%.5f, TrainMSE:%.5f, TrainR2:%.5f;" % (omit_dimension, train_auc, train_accuracy, train_mse, train_r2))
            print("Omit:%d, Test@Auc:%.5f, Test@Acy:%.5f, Test@MSE:%.5f, Test@R2:%.5f;" % (omit_dimension, test_auc, test_accuracy, test_mse, test_r2))

            print(print_separater)

        else:
            time_recorded = time.time()
            for ei in ei_ranges:

                # evaluate
                test_accuracy, test_mse, test_auc, test_r2 = \
                    evaluation(batch_size=args.batch_size,
                               dimension_count = args.dimension_count,
                               sess=sess,
                               logit_op=test_probability_eval, test_probability_op=test_probability_eval, true_label_op=test_true_label_tensor,
                               prdt_op=test_prdt_label_tensor, true_op=test_true_label_tensor, mse_op=test_mse_tensor, cort_op=test_correct_prdt_tensor,
                               actual_length=train_data_length,
                               iteration_for_each_epoch=iteration_for_each_epoch_test)
                train_accuracy, train_mse, train_auc,train_r2 = \
                    evaluation(batch_size=args.batch_size,
                               dimension_count=args.dimension_count,
                               sess=sess,
                               logit_op=train_probability_eval, test_probability_op=train_probability_eval,true_label_op=train_true_label_tensor,
                               prdt_op=train_prdt_label_tensor, true_op=train_true_label_tensor, mse_op=train_mse_tensor, cort_op=train_correct_prdt_tensor,
                               actual_length=train_data_length,
                               iteration_for_each_epoch=iteration_for_each_epoch_train)



                print("Epoch:%d, TrainAuc:%.5f, TrainAcy:%.5f, TrainMSE:%.5f, TrainR2:%.5f;" % (ei, train_auc, train_accuracy, train_mse, train_r2))
                print("Epoch:%d, Test@Auc:%.5f, Test@Acy:%.5f, Test@MSE:%.5f, Test@R2:%.5f;" % (ei, test_auc, test_accuracy, test_mse, test_r2))
                if time.time()-time_recorded>args.summary_seconds or ei==ei_start:
                    time_recorded=time.time()
                    print(print_test_info)
                print(print_separater)

                if ei == 15 or \
                        test_auc > previous_highest_auc and test_accuracy > previous_highest_accuracy and test_r2 > previous_highest_r2 and ei > 15:
                    if test_auc > previous_highest_auc:
                        previous_highest_auc = test_auc
                        print("Best AUC Found!")
                    if test_accuracy > previous_highest_accuracy:
                        previous_highest_accuracy = test_accuracy
                        print("Best Accuracy Found!")
                    if test_r2 > previous_highest_r2:
                        previous_highest_r2 = test_r2
                        print("Best R2 Found!")
                    if test_mse < previous_lowest_mse:
                        previous_lowest_mse = test_mse
                        print("Lowest MSE Found!")

                    previous_highest_auc_at_epoch = ei
                    print_test_info = "Epoch:%d, HighestAuc:%.5f, Accuracy:%.5f, MSE:%.5f, R2:%.5f" % (
                        ei, previous_highest_auc, test_accuracy, test_mse, test_r2)

                    print(print_test_info)
                    model_save_path = os.path.join(current_model_save_path, 'Model')
                    framework_save_path = os.path.join(current_model_save_path, 'Framework')
                    saver_full_model.save(sess, os.path.join(model_save_path, 'model'),
                                          global_step=global_step.eval(session=sess))
                    famework_saver.save(sess, os.path.join(framework_save_path, 'framework'),
                                        global_step=global_step.eval(session=sess))
                    local_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                    print('%s, ModelSaved@Epoch:%d' % (local_time, ei))
                    print(print_separater)

                for bid in range(iteration_for_each_epoch_train):
                    # train_batch_data, train_batch_label = \
                    #     sess.run([train_data_tensor, train_true_label_tensor])
                    # test_batch_data, test_batch_label = \
                    #     sess.run([test_data_tensor, test_true_label_tensor])

                    _ = sess.run(optimizer)

                summary_mark = time.time()
                current_train_loss_summary, current_train_loss_value = \
                    sess.run([merged_loss_summary, merged_loss_value])
                current_learning_rate_summary = sess.run(learning_rate_summary)

                current_train_performance_summary, current_train_performance_value = \
                    sess.run([merged_train_performance_summary, merged_train_performance_value])

                current_test_performance_summary, current_test_performance_value = \
                    sess.run([merged_test_performance_summary, merged_test_performance_value])

                # current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                # print('Training@%s: Epoch:%d/%d, Iter:%d/%d' % (current_time,ei,args.epochs,bid,iteration_for_each_epoch_train))
                # print("TrainLosses: CE:%.5f, WD:%.5f" %
                #       (current_train_loss_value[0],current_train_loss_value[1]))
                # print("TrainPerformance: Classification:%.5f, MSE:%.5f" %
                #       (current_train_performance_value[0], current_train_performance_value[1]))
                # print("TestPerformance: Classification:%.5f, MSE:%.5f" %
                #       (current_test_performance_value[0], current_test_performance_value[1]))
                # print(print_separater)
                # print(print_test_info)
                # print(print_separater)

                summary_writer.add_summary(current_train_loss_summary, global_step.eval(session=sess))
                summary_writer.add_summary(current_train_performance_summary, global_step.eval(session=sess))
                summary_writer.add_summary(current_test_performance_summary, global_step.eval(session=sess))
                summary_writer.add_summary(current_learning_rate_summary, global_step.eval(session=sess))
                summary_writer.flush()

                sess.run(epoch_step_increase_one_op)










def get_ordered_batch_data(batch_size,features,values,prev_end_index):
    if prev_end_index==-1:
        curt_start_index=0
    else:
        curt_start_index=prev_end_index
    curt_end_index=curt_start_index+batch_size
    if curt_end_index<=len(values)-1:
        batch_features=features[curt_start_index:curt_end_index,:]
        batch_values = values[curt_start_index:curt_end_index]
    else:
        batch_features_1 = features[curt_start_index:, :]
        batch_values_1 = values[curt_start_index:]
        still_need_num = batch_size-len(batch_values_1)
        batch_features_2 = features[0:still_need_num, :]
        batch_values_2 = values[0:still_need_num,:]
        batch_features=np.concatenate([batch_features_1,batch_features_2],axis=0)
        batch_values = np.concatenate([batch_values_1, batch_values_2], axis=0)

    return batch_features, batch_values, curt_end_index


def find_bn_avg_var(var_list):
    var_list_new=list()
    for ii in var_list:
        var_list_new.append(ii)


    all_vars = tf.global_variables()
    bn_var_list = [var for var in var_list if 'bn' in var.name]
    output_avg_var = list()
    for bn_var in bn_var_list:
        if 'gamma' in bn_var.name:
            continue
        bn_var_name = bn_var.name
        variance_name = bn_var_name.replace('beta','moving_variance')
        average_name = bn_var_name.replace('beta','moving_mean')
        variance_var = [var for var in all_vars if variance_name in var.name][0]
        average_var = [var for var in all_vars if average_name in var.name][0]
        output_avg_var.append(variance_var)
        output_avg_var.append(average_var)

    var_list_new.extend(output_avg_var)

    output=list()
    for ii in var_list_new:
        if ii not in output:
            output.append(ii)

    return output


def evaluation(batch_size,
               dimension_count,
               sess,
               test_probability_op, true_label_op,
               logit_op, prdt_op, true_op, mse_op, cort_op, iteration_for_each_epoch,
               actual_length, omit=-1):
    full_correct_test = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                                 dtype=np.bool)
    full_predict_test = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                                 dtype=np.float32)
    full_predict_logits = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                                   dtype=np.float32)
    full_true_test = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                              dtype=np.float32)
    full_mse_test = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                             dtype=np.float32)

    full_test_probability = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                                     dtype=np.float32)
    full_train_label = np.zeros(shape=[batch_size * iteration_for_each_epoch, 1],
                                dtype=np.float32)

    for bid in range(iteration_for_each_epoch):
        currect_prdt_logits, currect_prdt_label, current_true_label, current_correct, current_mse, current_test_probability, current_train_label = \
            sess.run([logit_op, prdt_op, true_op, cort_op, mse_op, test_probability_op, true_label_op])
        full_correct_test[bid * batch_size:(bid + 1) * batch_size] = current_correct
        full_predict_test[bid * batch_size:(bid + 1) * batch_size] = currect_prdt_label
        full_predict_logits[bid * batch_size:(bid + 1) * batch_size] = currect_prdt_logits
        full_true_test[bid * batch_size:(bid + 1) * batch_size] = current_true_label
        full_mse_test[bid * batch_size: (bid + 1) * batch_size] = current_mse
        full_test_probability[bid * batch_size: (bid + 1) * batch_size] = current_test_probability
        full_train_label[bid * batch_size: (bid + 1) * batch_size] = current_train_label



    full_correct_test = full_correct_test[0:actual_length]
    full_predict_test = full_predict_test[0:actual_length]
    full_true_test = full_true_test[0:actual_length]
    full_mse_test = full_mse_test[0:actual_length]
    full_predict_logits = full_predict_logits[0:actual_length]
    full_correct_test = [ii for ii in full_correct_test if ii == True]
    test_accuracy = float(len(full_correct_test)) / float(len(full_predict_test)) * 100
    test_mse = np.mean(full_mse_test)
    test_auc = roc_auc_score(y_true=full_true_test,
                             y_score=full_predict_logits)

    full_train_label = full_train_label[0:actual_length]
    full_test_probability = full_test_probability[0:actual_length]
    full_train_mean = np.mean(full_train_label)
    full_probability_mean = np.mean(full_test_probability)
    ssr = np.sum(np.power(full_test_probability-full_train_mean,2))
    sst = np.sum(np.power(full_train_label-full_train_mean,2)) + eps
    number_of_data = len(full_train_label)
    test_r2 = 1 - ((ssr/(number_of_data-dimension_count-1)) / (sst/(number_of_data-1)))

    current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
    # print('Epoch:%d, %s: %sAccuracy:%.5f, %sMse:%.5f, %sAuc:%.5f' %
    #       (ei, current_time, name_prefix, test_accuracy, name_prefix, test_mse, name_prefix, test_auc))


    return test_accuracy, test_mse, test_auc, test_r2