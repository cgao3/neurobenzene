import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

from commons.definitions import BuildInputTensor
import numpy as np
from utils.hexutils import MoveConvert

def run(const_graph_path, input_tensor_name='x_9x9_node:0', run_tensor_name='softmax_logits:0',
        with_batch_normalization=False, boardsize=9):
    output_graph_def=graph_pb2.GraphDef()
    with open(const_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(output_graph_def, name="")
    
    with tf.Session() as sess:
        input_node=sess.graph.get_tensor_by_name(input_tensor_name)
        run_node=sess.graph.get_tensor_by_name(run_tensor_name)

        print(input_node)
        print(run_node)
        input_builder=BuildInputTensor()
        batch_tensors=np.ndarray(shape=(1, boardsize+2, boardsize+2, 12))
        batch_tensors.fill(0)
        input_builder.set_position_tensors_in_batch(batch_tensors, kth=0, intMoveSeq=[])
        if with_batch_normalization:
            is_train_node = sess.graph.get_tensor_by_name('is_training')
            res=sess.run(run_node, feed_dict={input_node:batch_tensors, is_train_node:False})
        else:
            res=sess.run(run_node, feed_dict={input_node:batch_tensors})
        print("res: ", res)
        int_move=np.argmax(res)
        print('prediction: ', MoveConvert.int_move_to_raw(int_move, boardsize))
        print('argmax', np.argmax(res))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--boardsize', type=int, default=9, help='')
    parser.add_argument('--const_graph', type=str, default='', help='path to constant graph')
    args=parser.parse_args()

    run(args.const_graph)
