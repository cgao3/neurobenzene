import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

import numpy as np
from utils.hexutils import MoveConvert
from play.gtpinterface import GTPInterface
import sys
from commons.definitions import BuildInputTensor, INPUT_DEPTH
import math

'''
Neural net agent, load from freezed constant graph.
input: x_8x8_node:0 (optional is_training_node:0 set to False for batch normalization)
output: logits_8x8_node:0
'''
class NeuralNetAgent(object):
    def __init__(self, const_graph_path, name, boardsize, verbose=False, is_value_net=False):
        self.const_graph_path=const_graph_path
        self.boardsize = boardsize
        self.name = name
        self.is_value_net=is_value_net
        self._initialize_game([])
        self.verbose=verbose

    def reinitialize(self):
        self._initialize_game([])

    def _initialize_game(self, init_raw_move_seq):
        turn = 0
        self.int_game_state=[]
        self.black_int_moves=[]
        self.white_int_moves=[]
        for raw_move in init_raw_move_seq:
            int_move = MoveConvert.raw_move_to_int_move(raw_move, self.boardsize)
            if turn == 0:
                self.black_int_moves.append(int_move)
            else:
                self.white_int_moves.append(int_move)
            turn = (turn + 1) % 2
            self.int_game_state.append(int_move)

        self.input_tensor_builder = BuildInputTensor(self.boardsize)
        self.input_tensor = np.ndarray(dtype=np.float32, shape=(1, self.boardsize + 2, self.boardsize + 2, INPUT_DEPTH))
        self.input_tensor.fill(0)
        self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, self.int_game_state)
        self._load_model(self.const_graph_path)

    def _load_model(self, const_graph_path):

        graph_def = graph_pb2.GraphDef()
        with open(const_graph_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            _ = importer.import_graph_def(graph_def, name='')

        if not hasattr(self, 'sess'):
            self.sess=tf.Session()

        x_name='x_'+repr(self.boardsize)+'x'+repr(self.boardsize)+'_node:0'
        self.x_input_node=self.sess.graph.get_tensor_by_name(x_name)
        #print('find ', x_name, self.x_input_node)

        output_name='logits_'+repr(self.boardsize)+'x'+repr(self.boardsize)+'_node:0'
        self.logits_node=self.sess.graph.get_tensor_by_name(output_name)
        #print('find, ', output_name, self.logits_node)


    def set_boardsize(self, boardsize):
        self.boardsize=boardsize
        self._initialize_game([])

    ''' player should be either black or white, raw_move is something like a5, b7'''
    def play_move(self, player, raw_move):
        self.input_tensor.fill(0.0)
        int_move=MoveConvert.raw_move_to_int_move(raw_move, boardsize=self.boardsize)
        if player.lower()[0]=='b':
            self.black_int_moves.append(int_move)
        else:
            self.white_int_moves.append(int_move)
        self.int_game_state.append(int_move)
        return True

    def generate_move(self, player=None):
        if self.is_value_net:
            pass
        else:
            self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, self.int_game_state)
            if not hasattr(self, 'is_training_node') or not self.is_training_node:
                logits_score = self.sess.run(self.logits_node, feed_dict={self.x_input_node: self.input_tensor})
            else:
                print('not used is_training_node')
                logits_score = self.sess.run(self.logits_node, feed_dict={self.x_input_node:self.input_tensor,
                                                                                self.is_training_node:False})
            #print('logits score:', logits_score)
            empty_points=[]
            for i in range(self.boardsize*self.boardsize):
                if i not in self.int_game_state:
                    empty_points.append(i)
            selected_int_move=softmax_selection(logits_score, empty_points)

            self.int_game_state.append(selected_int_move)
            if player.lower()[0]=='b':
                self.black_int_moves.append(selected_int_move)
            else:
                self.white_int_moves.append(selected_int_move)

            raw_move=MoveConvert.int_move_to_raw(selected_int_move, boardsize=self.boardsize)
            assert (ord('a') <= ord(raw_move[0]) <= ord('z') and 0 <= int(raw_move[1:]) < self.boardsize ** 2)
            return raw_move

    def close_all(self):
        self.sess.close()


def softmax_selection(logits, empty_points, temperature=1.0):
    logits = np.squeeze(logits)
    #print(logits)
    #print('len(logits)', len(logits))

    effective_logits = [logits[i] for i in empty_points]
    max_value = np.max(effective_logits)
    effective_logits -=  max_value

    for i in range(len(effective_logits)):
        effective_logits[i]=effective_logits[i]/temperature

    effective_logits = np.exp(effective_logits)

    sum_value = np.sum(effective_logits)
    prob = effective_logits / sum_value
    boardsize=int(math.sqrt(len(prob)))
    print('boardsize:',boardsize)
    for i in range(len(prob)):
        x=i//boardsize
        y=i%boardsize
        print(repr(chr(ord('a')+x))+repr(y+1)+': '+repr(prob[i]))
    v = np.random.random()
    sum_v = 0.0
    action = None
    for i, e in enumerate(prob):
        sum_v += e
        if (sum_v >= v):
            action = i
            break
    ret = empty_points[action]
    return ret


def run2(const_graph_path, boardsize, verbose=False, is_value_net=False):
    #print('const_graph_path, ', const_graph_path)
    agent=NeuralNetAgent(const_graph_path, boardsize=boardsize, name='nn_agent',
            verbose=verbose, is_value_net=is_value_net)
    interface=GTPInterface(agent)
    while True:
        command=raw_input()
        success, response =interface.send_command(command)
        print("= " if success else "? ")
        print(str(response) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    import os
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_const_graph', type=str, default='', help="the path of the freezed constant graph")
    parser.add_argument('--value_net', action='store_true', default=False, help="value_net model?")
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose?')
    parser.add_argument('--boardsize', type=int, default=9)
    args=parser.parse_args()
    if not os.path.isfile(args.input_const_graph):
        print('please indicate path to constant graph, use --help for details')
        sys.exit(0)
    run2(const_graph_path=args.input_const_graph, boardsize=args.boardsize, 
            verbose=args.verbose, is_value_net=args.value_net)
