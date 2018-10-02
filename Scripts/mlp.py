# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import sys
sys.path.append('..')
from train_test_procedures import train_procedures
import argparse



input_args = ['--exp_name','Exp20180930',
              '--train_data_path','../Data/LinRui/TrainData.csv',
              '--test_data_path','../Data/LinRui/TestData.csv',
              '--model_save_path','../../FcSavedModel',
              '--log_path', '../../FcLogs',
              '--training_mode','0',

              '--batch_size','16',

              '--init_lr','0.0001',
              '--epochs','10000',
              '--summary_seconds','300',
              '--noise_level_for_continous_dimension','0.05',
              '--noise_level_for_discrete_dimension','0.3',
              '--omit_dimension_idx','',
              '--layer_channel_list','16',
              '--weight_decay_rate','0.004',


              '--train_num','107465',
              '--test_num', '18346',
              '--feature_dimension','25']

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--train_data_path', dest='train_data_path',type=str,required=True)
parser.add_argument('--test_data_path', dest='test_data_path',type=str,required=True)
parser.add_argument('--model_save_path', dest='model_save_path',type=str,required=True)
parser.add_argument('--log_path', dest='log_path',type=str,required=True)
parser.add_argument('--exp_name', dest='exp_name',type=str,required=True)


parser.add_argument('--feature_dimension', dest='feature_dimension',type=int,required=True)
parser.add_argument('--train_num', dest='train_num',type=int,required=True)
parser.add_argument('--test_num', dest='test_num',type=int,required=True)
parser.add_argument('--omit_dimension_idx', dest='omit_dimension_idx',type=str,required=True)



parser.add_argument('--batch_size', dest='batch_size',type=int,required=True)
parser.add_argument('--training_mode', dest='training_mode',type=int,required=True)
parser.add_argument('--init_lr', dest='init_lr',type=float,required=True)
parser.add_argument('--epochs', dest='epochs',type=int,required=True)
parser.add_argument('--summary_seconds', dest='summary_seconds',type=float,required=True)
parser.add_argument('--noise_level_for_continous_dimension', dest='noise_level_for_continous_dimension',type=float,required=True)
parser.add_argument('--noise_level_for_discrete_dimension', dest='noise_level_for_discrete_dimension',type=float,required=True)
parser.add_argument('--layer_channel_list', dest='layer_channel_list',type=str,required=True)
parser.add_argument('--weight_decay_rate', dest='weight_decay_rate',type=float,required=True)


def main(_):
    train_procedures(args=args)

#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
