import tensorflow as tf

from tensorflow.python.tools import freeze_graph

checkpoint_state_name = "checkpoint"
input_graph_name = "SaveFiles/Graph.pb"
output_graph_name = "SaveFiles/frozen_graph.pb"
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = "SaveFiles/model.ckpt-9999"
output_node_names = "Layer2/Output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
clear_devices = False

MIN_BOARD=5
MAX_BOARD=19

#freeze_graph.freeze_graph(input_graph_name, input_saver_def_path,
# input_binary, input_checkpoint_path, output_node_names, restore_op_name, filename_tensor_name, output_graph_name, clear_devices,"")

def freeze(input_graph, input_checkpoint, is_binary=False, output_node_names='x_9x9_node,output_layer/logits', output_graph='costant_graph.pb'):
    freeze_graph.freeze_graph(input_graph=input_graph, input_saver='', input_binary=is_binary, input_checkpoint=input_checkpoint,
                              output_node_names=output_node_names, restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                              output_graph=output_graph, clear_devices=False, initializer_nodes='')

if __name__ == "__main__":
    #output_node_names="x_8x8_node,x_9x9_node,x_10x10_node,x_11x11_node,x_12x12_node,x_13x13_node," + "logits_8x8_node,logits_9x9_node,logits_10x10_node," + "logits_11x11_node,logits_12x12_node,logits_13x13_node" 
    output_node_names=""
    for sz in range(MIN_BOARD, MAX_BOARD+1):
        output_node_names=output_node_names+"x_"+repr(sz)+"x"+repr(sz)+"_node,"
    for sz in range(MIN_BOARD, MAX_BOARD+1):
        if(sz==MAX_BOARD):
            output_node_names=output_node_names+"logits_"+repr(sz)+"x"+repr(sz)+"_node"
        else: 
            output_node_names=output_node_names+"logits_"+repr(sz)+"x"+repr(sz)+"_node,"
    import argparse
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_graph', type=str, default='', help='path to trained graph .pb')
    #parser.add_argument('--output', type=str, default='./output_constant.pb', help='output constant graph path')
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('--binary', action='store_true', default=False, help='is the input_graph binary?')
    parser.add_argument('--output_node_names', type=str, default=output_node_names, help='names in the input_graph needs to appear in output')
    #parser.add_argument('--with_batch_norm', action='store_true', default=False, help='with batch norm?')
    parser.add_argument('--with_v_and_q', action='store_true', default=True, help='with scalar v_node and vector q node')
    args=parser.parse_args()

    #if args.with_batch_norm:
    #    output_node_names += ',is_training'

    if args.with_v_and_q:
        for s in range(MIN_BOARD, MAX_BOARD+1, 1):
            output_node_names = output_node_names + ","+repr(s)+"x"+repr(s)+"_value"
        for s in range(MIN_BOARD, MAX_BOARD+1, 1):
            output_node_names = output_node_names + ","+repr(s)+"x"+repr(s)+"_q_values"
        #output_node_names +=",valid_to_play_points"

    print("default output_node_names:",output_node_names)
    import os
    if not os.path.isfile(args.input_graph):
        print('need to indicate input_graph, default txt format')
        exit(1)
    output=args.checkpoint+".const.pb"
    freeze(args.input_graph, args.checkpoint, args.binary, output_node_names, output)
    print('output constant graph is:', output)
