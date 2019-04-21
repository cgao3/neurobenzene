import tensorflow as tf
import os
import math
from commons.definitions import INPUT_DEPTH

MIN_BOARDSIZE = 8
MAX_BOARDSIZE = 15

epsilon = 0.001


class PlainCNNBatchNorm(object):
    '''
    Move prediction neural net,
    convlution size 3x3, 64 => last layer uses 1x1, 1 convolution -> reshape into a vector [boardsize x boardsize]
    input in the shape [batch, boardsize+2, boardsize+2, 12]

    ---- Naming ---
    input: x_8x8_node or x_9x9_node or x_10x10_node -> x_9x9_node:0
    y_star_node -> y_star_node:0

    output:
    logits_8x8_node
    logits_9x9_node ...
    logits_13x13_node

    accuracy evaluation: accuracy_evaluate_node -> accuracy_evaluate_node:0
    train op: train_op_node -> train_op_node:0
    '''

    def __init__(self, n_hiddenLayers=12):

        self.num_hidden_layers = n_hiddenLayers

        self.num_filters = 128
        self.filter_size = 3
        # 3x3 filter
        self.resue = False

    def batch_norm_wrapper(self, inputs, var_name_prefix, is_training_phase=True):
        pop_mean = tf.get_variable(name=var_name_prefix + '_pop_mean',
                                   shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable(name=var_name_prefix + '_pop_var',
                                  shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)

        gamma = tf.get_variable(name=var_name_prefix + '_gamma_batch_norm',
                                shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(1.0, tf.float32))
        beta = tf.get_variable(name=var_name_prefix + '_beta_batch_norm',
                               shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0, tf.float32))

        if is_training_phase:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * 0.999 + batch_mean * (1 - 0.999))
            train_var = tf.assign(pop_var, pop_var * 0.999 + batch_var * (1 - 0.999))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

    def build_graph(self, is_training_phase=True):
        x_node_dict = {}
        for i in range(MIN_BOARDSIZE, MAX_BOARDSIZE + 1, 1):
            name = "x_" + repr(i) + 'x' + repr(i) + "_node"
            x_node_dict[i] = tf.placeholder(dtype=tf.float32, shape=[None, i + 2, i + 2, INPUT_DEPTH], name=name)

        out_logits_dict = {}

        reuse = False
        for boardsize in range(MIN_BOARDSIZE, MAX_BOARDSIZE + 1, 1):
            with tf.variable_scope('plaincnn_batch_norm', reuse=reuse):
                init_stddev = math.sqrt(2.0 / (INPUT_DEPTH * self.filter_size * self.filter_size))
                w = tf.get_variable(name="weight1", shape=[self.filter_size, self.filter_size, INPUT_DEPTH, self.num_filters],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
                l = tf.nn.conv2d(x_node_dict[boardsize], w, strides=[1, 1, 1, 1], padding='VALID')
                l = self.batch_norm_wrapper(l, var_name_prefix='batch_norm1', is_training_phase=is_training_phase)
                l = tf.nn.relu(l)

                for i in range(self.num_hidden_layers - 1):
                    init_stddev = math.sqrt(2.0 / (self.num_filters * self.filter_size * self.filter_size))
                    w = tf.get_variable(name="weight%d"%(i+2), shape=[self.filter_size, self.filter_size, self.num_filters, self.num_filters],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
                    l = tf.nn.conv2d(l, w, strides=[1, 1, 1, 1], padding='SAME')
                    l = self.batch_norm_wrapper(l, var_name_prefix='batch_norm%d' % (i + 2), is_training_phase=is_training_phase)
                    l = tf.nn.relu(l)

                init_stddev = math.sqrt(2.0 / self.num_filters)
                w = tf.get_variable(name='weight_last_layer', shape=[1, 1, self.num_filters, 1],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
                l = tf.nn.conv2d(l, w, strides=[1, 1, 1, 1], padding='SAME')
            out_name = 'logits_' + repr(boardsize) + 'x' + repr(boardsize) + "_node"
            logits = tf.reshape(l, shape=[-1, boardsize * boardsize], name=out_name)
            out_logits_dict[boardsize] = logits
            reuse = True

        return x_node_dict, out_logits_dict

    def evaluate_on_test_data(self, input_data_file, boardsize, batch_size, saved_checkpoint, topk=1):

        y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        x_node_dict, out_logits_dict = self.build_graph(is_training_phase=False)

        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE
        eval_logits = out_logits_dict[boardsize]

        accuracy_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=eval_logits, targets=y_star, k=topk), tf.float32))
        from utils.input_data_util import PositionActionDataReader

        position_reader = PositionActionDataReader(position_action_filename=input_data_file, batch_size=batch_size, boardsize=boardsize)
        position_reader.enableRandomFlip = False

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, saved_checkpoint)
            batch_no = 0
            over_all_acc = 0.0
            while True:
                is_next_epoch = position_reader.prepare_next_batch()
                acc = sess.run(accuracy_op, feed_dict={
                    x_node_dict[boardsize]: position_reader.batch_positions, y_star: position_reader.batch_labels})
                print("batch no.: ", batch_no, " test accuracy: ", acc)
                batch_no += 1
                over_all_acc += acc
                if is_next_epoch:
                    break
            print("top: ", topk, " overall accuracy on test dataset", input_data_file, " is ", over_all_acc / batch_no)
            position_reader.close_file()

    '''
    needs to indicate what boardsize will be training on.
    '''

    def train(self, data_input_file, boardsize, batch_train_size,
              max_step, output_dir, resume_training=False, previous_checkpoint=''):
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE

        y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        x_node_dict, out_logits_dict = self.build_graph()
        train_logits = out_logits_dict[boardsize]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_star, logits=train_logits)

        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(
            y_star, tf.cast(tf.arg_max(train_logits, 1), tf.int32)), tf.float32), name='accuracy_node')

        optimizer = tf.train.AdamOptimizer().minimize(loss, name='train_op_node')

        from utils.input_data_util import PositionActionDataReader
        position_reader = PositionActionDataReader(position_action_filename=data_input_file,
                                                   batch_size=batch_train_size, boardsize=boardsize)
        position_reader.enableRandomFlip = True

        saver = tf.train.Saver(max_to_keep=20)
        acc_out_name = 'plaincnn_batch_norm_train_accuracies_' + repr(self.num_hidden_layers) \
                       + 'hidden_layers_' + repr(self.num_filters) + "_filters.txt"
        accu_writer = open(os.path.join(output_dir, acc_out_name), "w")

        epoch_acc_sum=0.0
        epoch_num=0
        eval_step=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if resume_training:
                saver.restore(sess, previous_checkpoint)

            for step in range(max_step + 1):
                is_next_epoch=position_reader.prepare_next_batch()
                if step % 50 == 0:
                    eval_step += 1
                    acc_train = sess.run(accuracy_op, feed_dict={
                        x_node_dict[boardsize]: position_reader.batch_positions, y_star: position_reader.batch_labels})
                    print("step: ", step, " train accuracy: ", acc_train)
                    accu_writer.write(repr(step) + ' ' + repr(acc_train) + '\n')
                    epoch_acc_sum +=acc_train

                if is_next_epoch:

                    print('epoch ', epoch_num, 'epoch train acc: ', epoch_acc_sum/eval_step)
                    accu_writer.write('epoch '+repr(epoch_num) + ' epoch_acc:' + repr(epoch_acc_sum/eval_step) + '\n')
                    epoch_num+=1
                    eval_step=0
                    epoch_acc_sum=0.0
                    saver.save(sess, os.path.join(output_dir, "plaincnn_batch_norm_model.ckpt"), global_step=epoch_num)

                sess.run(optimizer, feed_dict={x_node_dict[boardsize]: position_reader.batch_positions,
                                               y_star: position_reader.batch_labels})

        print("finished training on ", data_input_file, ", saving forward plaincnn with batch norm computation graph to " + output_dir)
        tf.reset_default_graph()
        self.build_graph(is_training_phase=False)
        tf.train.write_graph(tf.get_default_graph(), output_dir, 'plaincnn-batch-norm-graph.pbtxt', as_text=True)
        tf.train.write_graph(tf.get_default_graph(), output_dir, 'plaincnn-batch-norm-graph.pb', as_text=False)
        position_reader.close_file()
        accu_writer.close()
        print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=500)
    parser.add_argument('--batch_train_size', type=int, default=128)
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/')

    parser.add_argument('--resume_train', action='store_true', default=False)
    parser.add_argument('--previous_checkpoint', type=str, default='')

    parser.add_argument('--boardsize', type=int, default=9)
    parser.add_argument('--n_hidden_layer', type=int, default=6)

    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--topk', type=int, default=1)
    args = parser.parse_args()

    if args.evaluate:
        cnn = PlainCNNBatchNorm(n_hiddenLayers=args.n_hidden_layer)
        print('Testing')
        cnn.evaluate_on_test_data(args.input_file, boardsize=args.boardsize, batch_size=100, saved_checkpoint=args.previous_checkpoint,
                                  topk=args.topk)
        exit(0)

    if not os.path.isfile(args.input_file):
        print("please input valid path to input training data file")
        exit(0)

    if not os.path.isdir(args.output_dir):
        print("--output_dir must be a directory")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Training for board size", args.boardsize, args.boardsize)
    print("output directory: ", args.output_dir)

    cnn = PlainCNNBatchNorm(n_hiddenLayers=args.n_hidden_layer)

    cnn.train(data_input_file=args.input_file, boardsize=args.boardsize, batch_train_size=args.batch_train_size,
              max_step=args.max_train_step, output_dir=args.output_dir,
              resume_training=args.resume_train, previous_checkpoint=args.previous_checkpoint)
