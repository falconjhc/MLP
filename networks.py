import tensorflow as tf
from ops import batch_norm, lrelu, relu, fc, dropout
print_separater="#########################################################"



def network_model(features,
                  layer_channel_num_list,
                  device,
                  is_training,
                  initializer,
                  weight_decay,
                  weight_decay_rate,
                  reuse=False):

    network_name_prefix = "FullyConnected_%dLayers_Channels" % len(layer_channel_num_list)
    for ii in layer_channel_num_list:
        network_name_prefix = network_name_prefix+'_%d' % ii

    if is_training:
        print(print_separater)
        print("Training on %s" % network_name_prefix)
        print(print_separater)
        drop_v = 0.5
    else:
        drop_v = 0



    with tf.variable_scope(network_name_prefix):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        for ii in range(len(layer_channel_num_list)):
            if ii==0:
                input_feature = features
            else:
                input_feature = previous_output_feature
            if not ii == len(layer_channel_num_list)-1:
                current_output = dropout(relu(batch_norm(x=fc(x=input_feature,
                                                              output_size=layer_channel_num_list[ii],
                                                              weight_decay_rate=weight_decay_rate,
                                                              weight_decay=weight_decay,
                                                              parameter_update_device=device,
                                                              initializer=initializer,
                                                              scope='fc%d' % (ii+1),
                                                              name_prefix=network_name_prefix),
                                                         is_training=is_training,
                                                         scope='bn%d' % (ii+1),
                                                         parameter_update_device=device,
                                                         )),
                                         drop_v=drop_v)
            else:
                current_output = fc(x=input_feature,
                                    output_size=1,
                                    weight_decay_rate=weight_decay_rate,
                                    weight_decay=weight_decay,
                                    parameter_update_device=device,
                                    initializer=initializer,
                                    scope='fc%d' % (ii+1),
                                    name_prefix=network_name_prefix)
            previous_output_feature = current_output

        output = tf.nn.sigmoid(previous_output_feature)
        return previous_output_feature, output, network_name_prefix