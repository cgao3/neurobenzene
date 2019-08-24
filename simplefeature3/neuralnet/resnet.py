from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import tensorflow as tf
from commons.definitions import INPUT_DEPTH
from utils.input_data_util import PositionActionDataReader, PositionValueReader

'''
Residual neural network,
architecture specification:

input for 9x9 Hex board is [None, 11, 11, 12]
=> conv3x3, num_filters, output is [9,9,32]

=> one resnet block:
BN -> ReLU -> conv3x3, num_filters ->
BN -> ReLU -> conv3x3, num_filters ->
addition with x_i

=> k resnet blcoks repetition

naming:
x_nxn_node: where n is board size
y_star_node:

is_training_node:

logits_nxn_node: where n is boardsize
'''

epsilon = 0.001

MIN_BOARDSIZE=6
MAX_BOARDSIZE=19

class ResNet(object):
    def __init__(self, num_blocks=10, num_filters=64, fc_policy_head=False, fc_q_head=False):

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.fc_policy_head=fc_policy_head
        self.fc_q_head=fc_q_head

        self.var_dict={}

    def batch_norm_wrapper(self, inputs, var_name_prefix, is_training_phase=True):
        pop_mean = tf.get_variable(name=var_name_prefix + '_pop_mean',
                                   shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable(name=var_name_prefix + '_pop_var',
                                  shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)

        gamma = tf.get_variable(name=var_name_prefix + '_gamma_batch_norm',
                                shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(1.0, tf.float32))
        beta = tf.get_variable(name=var_name_prefix + '_beta_batch_norm',
                               shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0, tf.float32))

        self.var_dict[pop_mean.op.name]=pop_mean
        self.var_dict[pop_var.op.name]=pop_var
        self.var_dict[gamma.op.name]=gamma
        self.var_dict[beta.op.name]=beta

        if is_training_phase:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * 0.999 + batch_mean * (1 - 0.999))
            train_var = tf.assign(pop_var, pop_var * 0.999 + batch_var * (1 - 0.999))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

    '''
    build a graph for all input board sizes,
    :return all input node, all output node
    a three 'heads' neural net
    head one: a vector of q values
    head two: a single v value
    head three: move probabilities
    '''
    def build_graph(self, is_training_phase=True):
        x_node_dict = {}
        empty_points_mask_dict = {}
        self.w_list=[]
        for i in range(MIN_BOARDSIZE, MAX_BOARDSIZE + 1, 1):
            name = "x_" + repr(i) + 'x' + repr(i) + "_node"
            x_node_dict[i] = tf.placeholder(dtype=tf.float32, shape=[None, i + 2, i + 2, INPUT_DEPTH], name=name)
            empty_points_mask_dict[i] = tf.placeholder(dtype=tf.bool, shape=[None, i * i], name='valid_to_play_points')

        policy_logits_dict = {}
        q_values_dict = {}
        value_dict = {}

        reuse=False
        for boardsize in range(MIN_BOARDSIZE, MAX_BOARDSIZE+1, 1):
            with tf.variable_scope('resnet', reuse=reuse):
                w1 = tf.get_variable(name="w1", shape=[3, 3, INPUT_DEPTH, self.num_filters], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, math.sqrt(1.0 / (3 * 3 * INPUT_DEPTH))))
                h = tf.nn.conv2d(x_node_dict[boardsize], w1, strides=[1, 1, 1, 1], padding='VALID')
                self.w_list.append(w1)
                self.var_dict[w1.op.name]=w1
                for i in range(self.num_blocks):
                    h = self._build_one_block(h, name_prefix='block%d' % i, is_training_phase=is_training_phase)

                h=self.batch_norm_wrapper(h, var_name_prefix='left_res_blocks/batch_norm1', is_training_phase=is_training_phase)
                h=tf.nn.relu(h)

                '''
                last layer uses 1x1,1 convolution, then reshape the output as [boardsize*boardsize]
                '''
                in_depth = h.get_shape()[-1]

                xavier = math.sqrt(2.0 / (1 * 1 * self.num_filters))
                w = tf.get_variable(dtype=tf.float32, name="weight", shape=[1, 1, in_depth, 1],
                                    initializer=tf.random_normal_initializer(stddev=xavier))
                b=tf.get_variable(dtype=tf.float32, name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
                h_p = tf.nn.conv2d(h, w, strides=[1, 1, 1, 1], padding='SAME') + b

                self.w_list.append(w)
                self.w_list.append(b)
                self.var_dict[w.op.name]=w
                self.var_dict[b.op.name]=b

                w_v = tf.get_variable(dtype=tf.float32, name="weight_vhead", shape=[1, 1, in_depth, 1],
                                      initializer=tf.random_normal_initializer(stddev=xavier))
                b_v = tf.get_variable(dtype=tf.float32, name='bias_vhead', shape=[1], initializer=tf.constant_initializer(0.0))
                h_v = tf.nn.conv2d(h, w_v, strides=[1, 1, 1, 1], padding='SAME') + b_v

                self.w_list.append(w_v)
                self.w_list.append(b_v)
                self.var_dict[w_v.op.name]=w_v
                self.var_dict[b_v.op.name]=b_v

                reuse = True
                out_name='logits_'+repr(boardsize)+'x'+repr(boardsize)+'_node'
                out_name_v='value_logits_'+repr(boardsize)+'x'+repr(boardsize)+'_node'
                output_v_node_name=repr(boardsize)+'x'+repr(boardsize)+'_value'
                output_q_node_name=repr(boardsize)+'x'+repr(boardsize)+'_q_values'

            if not self.fc_policy_head:
                policy_logits_dict[boardsize] = tf.reshape(h_p, shape=[-1, boardsize* boardsize], name=out_name)
            #add fc for policy output
            else:
                h_p = tf.reshape(h_p, shape=[-1, boardsize* boardsize])
                w_fc_p = tf.get_variable(dtype=tf.float32, name="weight_fc_p%d"%boardsize, shape=[boardsize * boardsize, boardsize * boardsize], initializer=tf.random_normal_initializer(stddev=0.1))
                self.w_list.append(w_fc_p)
                self.var_dict[w_fc_p.op.name]=w_fc_p
                b_fc_p = tf.get_variable(dtype=tf.float32, name='bias_fc_p%d'%boardsize, shape=[boardsize * boardsize], initializer=tf.constant_initializer(0.0))
                self.w_list.append(b_fc_p)
                self.var_dict[b_fc_p.op.name]=b_fc_p
                policy_logits_dict[boardsize] = tf.add(tf.matmul(h_p, w_fc_p), b_fc_p, name=out_name)

            #q and v estimates 
            h = tf.reshape(h_v, shape=[-1, boardsize*boardsize], name=out_name_v)
            if self.fc_q_head:
                w_fc_q = tf.get_variable(dtype=tf.float32, name="weight_q%d"%boardsize, shape=[boardsize * boardsize, boardsize * boardsize], initializer=tf.random_normal_initializer(stddev=0.1))
                self.w_list.append(w_fc_q)
                self.var_dict[w_fc_q.op.name]=w_fc_q
                b_fc_q = tf.get_variable(dtype=tf.float32, name='bias_q%d'%boardsize, shape=[boardsize * boardsize], initializer=tf.constant_initializer(0.0))
                self.w_list.append(b_fc_q)
                self.var_dict[b_fc_q.op.name]=b_fc_q
                q_values = tf.nn.tanh(tf.add(tf.matmul(h, w_fc_q), b_fc_q), name=output_q_node_name)
            else:
                q_values = tf.nn.tanh(h, name=output_q_node_name) #uncomment for without fc in q-value layer
            q_values_dict[boardsize]=q_values

            w_fc_v = tf.get_variable(dtype=tf.float32, name="weight_v%d"%boardsize, shape=[boardsize * boardsize, 1], initializer=tf.random_normal_initializer(stddev=0.1))
            b_fc_v = tf.get_variable(dtype=tf.float32, name='bias_v%d'%boardsize, shape=[1], initializer=tf.constant_initializer(0.0))

            self.w_list.append(w_fc_v)
            self.w_list.append(b_fc_v)
            self.var_dict[w_fc_v.op.name]=w_fc_v
            self.var_dict[b_fc_v.op.name]=b_fc_v
            v_value_original = tf.nn.tanh(tf.add(tf.matmul(h, w_fc_v), b_fc_v))
            v_value = tf.reshape(v_value_original, shape=(-1,), name=output_v_node_name)
            value_dict[boardsize] = v_value

        return x_node_dict, empty_points_mask_dict, policy_logits_dict, q_values_dict, value_dict

    def _build_one_block(self, inputs, name_prefix, is_training_phase=True):
        original_inputs = inputs
        b1 = self.batch_norm_wrapper(inputs, var_name_prefix=name_prefix + '/batch_norm1', is_training_phase=is_training_phase)
        b1_hat = tf.nn.relu(b1)

        in_block_w1 = tf.get_variable(name=name_prefix + '/weight1', shape=[3, 3, self.num_filters, self.num_filters],
                                      dtype=tf.float32, initializer=tf.random_normal_initializer(
                stddev=math.sqrt(1.0 / (9 * self.num_filters))))
        h1 = tf.nn.conv2d(b1_hat, in_block_w1, strides=[1, 1, 1, 1], padding='SAME')

        b2 = self.batch_norm_wrapper(h1, var_name_prefix=name_prefix + '/batch_norm2', is_training_phase=is_training_phase)
        b2_hat = tf.nn.relu(b2)
        in_block_w2 = tf.get_variable(name_prefix + '/weight2', shape=[3, 3, self.num_filters, self.num_filters],
                                      dtype=tf.float32, initializer=tf.random_normal_initializer(
                stddev=math.sqrt(1.0 / (9 * self.num_filters))))

        h2 = tf.nn.conv2d(b2_hat, in_block_w2, strides=[1, 1, 1, 1], padding='SAME')
        self.w_list.append(in_block_w1)
        self.w_list.append(in_block_w2)
        self.var_dict[in_block_w1.op.name]=in_block_w1
        self.var_dict[in_block_w2.op.name]=in_block_w2
        return tf.add(original_inputs, h2)

    def train(self, src_train_data_path, boardsize, hpr, output_dir, resume_training=False,
              previous_checkpoint=''):
        batch_train_size=hpr['batch_size']
        max_step=hpr['max_step']
        epoch_limit=hpr['epoch_limit']
        p_loss_weight = hpr['policy_loss_weight']
        v_loss_weight = hpr['value_loss_weight']
        penalty_loss_weight = hpr['value_penalty_weight']
        coeff_v = hpr['v_coeff']
        coeff_q = hpr['q_coeff']
        coeff_q_all = hpr['q_all_coeff']
        l2_loss_weight = hpr['l2_weight']
        using_quadratic=hpr['quadratic']
        verbose=hpr['verbose']
        labeltype=hpr['label_type']
        
        print(hpr)
        (x_node_dict, empty_points_mask_dict, policy_logits_dict, q_values_dict, value_dict) = self.build_graph(is_training_phase=True)
        y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        y_distr = tf.placeholder(dtype=tf.float32, shape=(None, boardsize*boardsize), name='y_distr_star')
        z_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='z_node')

        assert MIN_BOARDSIZE<= boardsize <= MAX_BOARDSIZE
        policy_logits=policy_logits_dict[boardsize]
        if labeltype=="exclusive": 
            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_star, logits=policy_logits)
        elif labeltype == "prob":
            #softlogits=tf.nn.softmax(logits=policy_logits)
            #policy_loss = - tf.reduce_sum(y_distr*tf.log(softlogits), -1)
            policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_distr), logits=policy_logits)
        else: 
            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_star, logits=policy_logits)
        #print("policy_loss:",policy_loss)
        policy_loss=tf.reduce_mean(policy_loss)
        one_hot_vec=tf.one_hot(y_star, boardsize*boardsize)
        mask=tf.cast(one_hot_vec, tf.bool)
        v_value = value_dict[boardsize]
        only_v_loss = tf.square(z_node - v_value)

        q_values = q_values_dict[boardsize]
        squared=tf.square(tf.expand_dims(z_node,-1)+q_values)

        #print("emptypoint shape:", empty_points_mask_dict[boardsize])
        all_q_a_error = tf.where(empty_points_mask_dict[boardsize], squared, squared-squared)
        #print("all_q_a_error:",all_q_a_error)

        and_loss = tf.maximum(-z_node, 0) * (tf.reduce_sum(all_q_a_error, -1)/tf.cast(tf.count_nonzero(all_q_a_error, 1),tf.float32))
        #print('one_hot_vec shape:', one_hot_vec.shape)
        q_a=tf.multiply(one_hot_vec, q_values)
        q_a=tf.reduce_sum(q_a, 1)
        batch_value_loss = coeff_v*only_v_loss + coeff_q*tf.square(z_node+ q_a) + coeff_q_all*and_loss

        #print('batch_value_loss', batch_value_loss)
        value_loss = tf.reduce_mean(batch_value_loss)
        only_v_loss_mean= tf.reduce_mean(only_v_loss)
        #print('value loss:', value_loss)

        valid_q_values=tf.where(empty_points_mask_dict[boardsize], q_values, 1.0+q_values)
        #print('valid_q_values:', valid_q_values)
        #print('q values:', q_values)
        min_q_value=tf.reduce_min(valid_q_values, axis=1)
        #print('min_q_value:', min_q_value)
        #penalty=tf.abs(min_q_value + v_value)
        penalty=tf.square(min_q_value+v_value)
        #print('penalty_batch:', penalty)
        penalty=tf.reduce_mean(penalty)
        #print('penalty', penalty)

        loss = p_loss_weight*policy_loss + v_loss_weight*value_loss + penalty_loss_weight*penalty
        regularizer = 0.0
        for w in self.w_list:
            regularizer +=tf.nn.l2_loss(w)
        loss = loss + l2_loss_weight*regularizer
        
        if hpr['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer().minimize(loss, name='train_op')
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate=0.0005, momentum=0.90).minimize(loss, name='train_op')
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, name='train_op')

        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(
            y_star, tf.cast(tf.argmax(policy_logits, 1), tf.int32)), tf.float32), name='accuracy_node')

        reader = PositionActionDataReader(position_action_filename=src_train_data_path,
                                          batch_size=batch_train_size, boardsize=boardsize, with_value=True)
        reader.enableRandomFlip = True
        saver = tf.train.Saver(max_to_keep=epoch_limit+1)
        acc_out_name='resnet_train_accuracies'+repr(self.num_blocks)+'_blocks_'+\
                     repr(self.num_filters)+'_filters_perconv.txt'
        accu_writer = open(os.path.join(output_dir, acc_out_name), "w")
        accu_writer.write('#'+str(hpr)+'\n')

        epoch_acc_sum = 0.0
        epoch_value_sum = 0.0
        epoch_num = 0
        eval_step = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver_restore = tf.train.Saver(var_list=self.var_dict)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if resume_training:
                if os.path.isfile(previous_checkpoint+".meta"):
                    try: 
                        saver_restore.restore(sess, previous_checkpoint)
                    except tf.errors.NotFoundError:
                        print('ignore variables failed to restore, since we have run init')
                        pass
                else:
                    print("resume failed, no such checkpoint;  start from random weights\n")

            for step in range(max_step + 1):
                if epoch_num>= epoch_limit: 
                    print("max epochs reached, stop")
                    break
                is_next_epoch=reader.prepare_next_batch()

                if step % 50 == 0:
                    eval_step += 1
                    acc_train, value_loss_train, only_v_loss_train = sess.run([accuracy_op, value_loss, only_v_loss_mean],
                                                           feed_dict={x_node_dict[boardsize]: reader.batch_positions, y_star: reader.batch_labels,
                                                               z_node: -reader.batch_q_sa_values, y_distr:reader.batch_targets,
                                                                      empty_points_mask_dict[boardsize]: reader.batch_empty_points})
                    accu_writer.write('#'+repr(step) + ' ' + repr(acc_train) + ' ' + repr(value_loss_train) + ' '+ repr(only_v_loss_train) +'\n')
                    if verbose: 
                        print("step: ", step, " resnet train accuracy: ", acc_train, 'total v loss:', value_loss_train, 'v loss:', only_v_loss_train)

                    epoch_acc_sum += acc_train
                    epoch_value_sum += value_loss_train

                if is_next_epoch:
                    if verbose and eval_step>=1: 
                        print('epoch ', epoch_num+1, 'epoch train acc: ', epoch_acc_sum/eval_step, 'epoch value loss:', epoch_value_sum/eval_step)
                        accu_writer.write('epoch '+repr(epoch_num+1) + ' epoch_acc and value loss:' + 
                            repr(epoch_acc_sum/eval_step) + ' ' + repr(epoch_value_sum/eval_step)+'\n')
                    epoch_num += 1
                    eval_step = 0
                    epoch_acc_sum = 0.0
                    epoch_value_sum = 0.0
                    saver.save(sess, os.path.join(output_dir,os.path.basename(src_train_data_path)+".ckpt"), global_step=epoch_num)

                sess.run(optimizer,
                         feed_dict={x_node_dict[boardsize]: reader.batch_positions, y_star: reader.batch_labels,
                             z_node:-reader.batch_q_sa_values, y_distr:reader.batch_targets,
                                    empty_points_mask_dict[boardsize]:reader.batch_empty_points})
            print("Training finished.")
            saver.save(sess, os.path.join(output_dir, os.path.basename(src_train_data_path)+".ckpt"), global_step=epoch_num)

        #print('reset the graph for inference batch normalization!') # can we do better?
        #tf.reset_default_graph()
        #self.build_graph(is_training_phase=False)
        #tf.train.write_graph(tf.get_default_graph(), output_dir, 'resnet.graphpbtxt', as_text=True)
        #tf.train.write_graph(tf.get_default_graph(), output_dir, 'resnet.graphpb', as_text=False)
        accu_writer.close()
        reader.close_file()
        print('Done.')

    '''
    Only evaluate the value head
    '''
    def evaluate_value_on_test_data(self, src_test_data, boardsize, hpr, saved_checkpoint):
        print('On dataset', src_test_data, ' testing')
        #topk=hpr['topk']
        x_node_dict, empty_points_mask_dict, policy_logits_dict, q_values_dict, value_dict = self.build_graph(is_training_phase=False)
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE
        #logits=policy_logits_dict[boardsize]
        batch_size = hpr['batch_size']
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE

        #y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        v_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='v_node')
        #q_sa_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='q_sa_node')
        v_value=value_dict[boardsize]
        batch_value_loss = tf.square(v_node - v_value)
        print('batch_value_loss', batch_value_loss)
        value_loss = tf.reduce_mean(batch_value_loss)
        reader = PositionValueReader(position_value_filename=src_test_data, batch_size=batch_size, boardsize=boardsize)
        reader.enableRandomFlip = False
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, saved_checkpoint)
            batch_no = 0
            overall_loss =0.0
            while True:
                is_next_epoch = reader.prepare_next_batch()
                loss_v = sess.run(value_loss, feed_dict={x_node_dict[boardsize]: reader.batch_positions, v_node:reader.batch_values})
                print("batch no: ", batch_no, 'value loss:', loss_v)
                batch_no += 1
                overall_loss += loss_v
                if is_next_epoch:
                    break
            print("overall value loss on test dataset", src_test_data, " is ", overall_loss / batch_no)
            print('saving resnet forward evaluation graph to ./')
            save_name="resnet_evaluate_value_"+repr(hpr['n_hidden_blocks'])+"_"+repr(hpr['n_filters_per_layer'])
            tf.train.write_graph(tf.get_default_graph(), './', save_name+'.pbtxt', as_text=True)
            tf.train.write_graph(tf.get_default_graph(), './', save_name+'.pb', as_text=False)
            reader.close_file()
    '''
    evaluate three heads
    '''
    def evaluate_on_test_data(self, src_test_data, boardsize, hpr, saved_checkpoint):
        print('On dataset', src_test_data, ' testing')
        topk=hpr['topk']
        x_node_dict, empty_points_mask_dict, policy_logits_dict, q_values_dict, value_dict = self.build_graph(is_training_phase=False)
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE
        logits=policy_logits_dict[boardsize]
        batch_size = hpr['batch_size']
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE

        y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        v_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='v_node')
        q_sa_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='q_sa_node')
        one_hot_vec = tf.one_hot(y_star, boardsize * boardsize)
        mask = tf.cast(one_hot_vec, tf.bool)
        v_value = value_dict[boardsize]
        q_values = q_values_dict[boardsize]
        alpha=0.0
        batch_value_loss = alpha*tf.square((v_node - v_value)) + (1.0-alpha)*tf.square(q_sa_node - tf.boolean_mask(q_values, mask))
        print('batch_value_loss', batch_value_loss)
        value_loss = tf.reduce_mean(batch_value_loss)

        accuracy_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=y_star, k=topk), tf.float32))
        reader = PositionActionDataReader(position_action_filename=src_test_data,
                                          batch_size=batch_size, boardsize=boardsize, with_value=True)
        reader.enableRandomFlip = False
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, saved_checkpoint)
            batch_no = 0
            overall_acc = 0.0
            overall_loss =0.0
            while True:
                is_next_epoch = reader.prepare_next_batch()
                acc, loss_v = sess.run([accuracy_op, value_loss],
                                       feed_dict={x_node_dict[boardsize]: reader.batch_positions, y_star: reader.batch_labels,
                                                  v_node: -reader.batch_q_sa_values, q_sa_node: reader.batch_q_sa_values,
                                                  empty_points_mask_dict[boardsize]: reader.batch_empty_points})
                print("batch no: ", batch_no, 'policy accuracy:', acc,'v head contrib:', alpha, 'value loss:', loss_v)
                batch_no += 1
                overall_acc += acc
                overall_loss += loss_v
                if is_next_epoch:
                    break
            print("top: ", topk, "overall accuracy on test dataset", src_test_data, " is ", overall_acc / batch_no)
            print("overall value loss on test dataset", src_test_data, " is ", overall_loss / batch_no)
            print('saving resnet forward evaluation graph to ./')
            save_name="resnet_evaluate_"+repr(hpr['n_hidden_blocks'])+"_"+repr(hpr['n_filters_per_layer'])
            tf.train.write_graph(tf.get_default_graph(), './', save_name+'.pbtxt', as_text=True)
            tf.train.write_graph(tf.get_default_graph(), './', save_name+'.pb', as_text=False)
            reader.close_file()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_train_step', type=int, default=500, help='max steps')
    parser.add_argument('--epoch_limit', type=int, default=10, help='max number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--input_file', type=str, default='', help='input game-state data set for training/testing')

    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/', help='where to put logs')
    parser.add_argument('--resume_train', action='store_true', default=False, help='resume from old checkpoint')
    parser.add_argument('--previous_checkpoint', type=str, default='', help='where is the old checkpoint')
    parser.add_argument('--evaluate', action='store_true', default=False, help='to perform test')
    parser.add_argument('--boardsize', type=int, default=13, help='board size')
    parser.add_argument('--topk', type=int, default=1, help='top k in testing mode')

    parser.add_argument('--policy_loss_weight', default=1.0, type=float, help='policy gradient weight in loss function')
    parser.add_argument('--value_loss_weight', default=0.1, type=float, help='value loss weight')
    parser.add_argument('--value_penalty_weight', default=0.25, type=float, help='value inconsistency weight')
    parser.add_argument('--evaluate_value', action='store_true', default=False, help='to perform test only value head')
    parser.add_argument('--l2_weight', type=float, default=0.0, help='l2 regularizer weight')
    parser.add_argument('--v_coeff', type=float, default=1.0, help='v coefficient')
    parser.add_argument('--q_coeff', type=float, default=1.0,  help='q coefficient')
    parser.add_argument('--q_all_coeff', type=float, default=1.0, help='augmented q coefficient')
    parser.add_argument('--quadratic_penalty', action='store_true', default=True, help='using quadratic penalty?')

    parser.add_argument('--n_hidden_blocks', type=int,  default=10, help='num of hidden blocks')
    parser.add_argument('--n_filters_per_layer', type=int,  default=32, help='num of filters per layer')
    parser.add_argument('--fc_policy_head', action='store_true',  default=False, help='fully-connected policy head? True for non-transferable policy head')
    parser.add_argument('--fc_q_head', action='store_true',  default=False, help='fully-connected action-value q head? True for non-transferable q head')
    parser.add_argument('--verbose', action='store_true',  default=False, help='verbose printout?')
    parser.add_argument('--label_type', type=str, default='exclusive', help='exclusive or prob')
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or momentum')

    args = parser.parse_args()

    hyperparameter = {}
    hyperparameter['batch_size']=args.batch_size
    hyperparameter['max_step']=args.max_train_step
    hyperparameter['policy_loss_weight']=args.policy_loss_weight
    hyperparameter['value_loss_weight']=args.value_loss_weight
    hyperparameter['value_penalty_weight']=args.value_penalty_weight
    hyperparameter['topk']=args.topk
    hyperparameter['v_coeff'] = args.v_coeff
    hyperparameter['q_coeff'] = args.q_coeff
    hyperparameter['q_all_coeff'] = args.q_all_coeff
    hyperparameter['l2_weight']=args.l2_weight
    hyperparameter['quadratic']=args.quadratic_penalty
    hyperparameter['epoch_limit']=args.epoch_limit

    n_hidden_blocks=args.n_hidden_blocks
    n_filters_per_layer=args.n_filters_per_layer
    hyperparameter['n_hidden_blocks']=n_hidden_blocks
    hyperparameter['n_filters_per_layer']=n_filters_per_layer
    hyperparameter['verbose']=args.verbose
    hyperparameter['label_type']=args.label_type
    hyperparameter['optimizer']=args.optimizer
    hyperparameter['fc_policy_head']=args.fc_policy_head
    hyperparameter['fc_q_head']=args.fc_q_head
    
    fc_policy_head=args.fc_policy_head
    fc_q_head=args.fc_q_head

    if args.evaluate:
        print('Testing')
        resnet=ResNet(num_blocks=n_hidden_blocks, num_filters=n_filters_per_layer, fc_policy_head=fc_policy_head)
        resnet.evaluate_on_test_data(args.input_file, args.boardsize, hpr=hyperparameter, saved_checkpoint=args.previous_checkpoint)
        exit(0)

    if args.evaluate_value:
        print('Only Value Testing')
        resnet=ResNet(num_blocks=n_hidden_blocks, num_filters=n_filters_per_layer, fc_policy_head=fc_policy_head)
        resnet.evaluate_value_on_test_data(args.input_file, args.boardsize, hpr=hyperparameter, saved_checkpoint=args.previous_checkpoint)
        exit(0)

    if not os.path.isfile(args.input_file):
        print("please input valid path to input training data file")
        exit(0)
    if not os.path.isdir(args.output_dir):
        print("--output_dir must be a directory")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Training for board size", args.boardsize)
    print("output directory: ", args.output_dir)
    resnet = ResNet(num_blocks=n_hidden_blocks, num_filters=n_filters_per_layer, fc_policy_head=fc_policy_head, fc_q_head=fc_q_head)
    resnet.train(src_train_data_path=args.input_file, boardsize=args.boardsize, hpr=hyperparameter, output_dir=args.output_dir,
                 resume_training=args.resume_train, previous_checkpoint=args.previous_checkpoint)
