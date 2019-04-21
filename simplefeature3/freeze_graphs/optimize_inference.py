
from tensorflow.python.tools import optimize_for_inference
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_io
from google.protobuf import text_format
from tensorflow.python.framework import dtypes
import os
import tensorflow as tf

if __name__ =='__main__':
    input_node_names=""
    for sz in range(8,20):
        input_node_names=input_node_names+"x_"+repr(sz)+"x"+repr(sz)+"_node"
        if sz !=19:
            input_node_names = input_node_names+","
    output_node_names=""
    for sz in range(8,20):
        if(sz==19):
            output_node_names=output_node_names+"logits_"+repr(sz)+"x"+repr(sz)+"_node"
        else: 
            output_node_names=output_node_names+"logits_"+repr(sz)+"x"+repr(sz)+"_node,"
    for s in range(8,20,1):
            output_node_names = output_node_names + ","+repr(s)+"x"+repr(s)+"_value"
    for s in range(8,20,1):
            output_node_names = output_node_names + ","+repr(s)+"x"+repr(s)+"_q_values"

    placeholder_type_enum=tf.float32.as_datatype_enum
    import argparse
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_graph', type=str, default='', help='text format input graph')
    #parser.add_argument('--output', type=str, default='optimized.pb', help='output of optimized graph')
    parser.add_argument('--frozen_graph', action='store_false', default=True, help='is frozen graph?')
    args=parser.parse_args()
    import sys
    if not gfile.Exists(args.input_graph):
        print("Input graph file '" + args.input_graph + "' does not exist!")
        sys.exit(0)

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(args.input_graph, "rb") as f:
        data = f.read()
        if args.frozen_graph:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data.decode("utf-8"), input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_node_names.split(","),
        output_node_names.split(","), placeholder_type_enum)

    output=args.input_graph+'_optimized'
    if args.frozen_graph:
        f = gfile.FastGFile(output, "w")
        f.write(output_graph_def.SerializeToString())
    else:
        graph_io.write_graph(output_graph_def,
                             os.path.dirname(output),
                             os.path.basename(output))

