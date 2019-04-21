#include<iostream>
#include<vector>
#include<utility>
#include <ConstBoard.hpp>
#include <cfloat>
#include <tensorflow/c/c_api.h>


#include "tf_utils.hpp"
#include "NeuroEvaluate.hpp"

typedef std::vector<float> v1d;
typedef std::vector<v1d> v2d;
typedef std::vector<v2d> v3d;
typedef std::vector<v3d> v4d;

NNEvaluator::~NNEvaluator(){
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(m_sess, status);
    if (TF_GetCode(status) != TF_OK) {
        m_sess=nullptr;
        m_graph=nullptr;
        std::cout<<"TF_CloseSession error in NNEvaluate deconstruction"<<std::endl;
        TF_CloseSession(m_sess, status);
        TF_DeleteSession(m_sess, status);
        TF_DeleteStatus(status);
        return;
    }
    TF_DeleteSession(m_sess, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout<<"TF_DeleteSession error in NNEvaluate deconstruction"<<std::endl;
    } 
    if(m_graph!=nullptr){
        TF_DeleteGraph(m_graph);
        m_graph=nullptr;
    }
    TF_DeleteStatus(status);
    m_sess = nullptr;
    m_graph = nullptr;
}

NNEvaluator::NNEvaluator(){
}

NNEvaluator::NNEvaluator(std::string model_path):m_sess(nullptr), m_graph(nullptr), 
    m_min_q_combine_weight(0.0), m_q_weight_to_p(0.0), m_product_propagate_weight(0.0) {
    load_nn_model(model_path);
}

void NNEvaluator::load_nn_model(std::string model_path){
    TF_Status* status = TF_NewStatus();
    if(m_sess!= nullptr) {
        TF_DeleteSession(m_sess, status);
        if (TF_GetCode(status) != TF_OK) {
            TF_DeleteStatus(status);
            std::cout<<"load_nn error in deleting existing session!\n";
            return;
        }
        m_sess = nullptr;
    }
    if(m_graph!=nullptr){
        TF_DeleteGraph(m_graph);
        m_graph=nullptr;
    }
    this->m_neural_model_path=model_path;
    TF_Graph* graph = tf_utils::LoadGraphDef(this->m_neural_model_path.c_str());
    this->m_graph=graph;
    TF_SessionOptions* opts = TF_NewSessionOptions();
    //TF_SetConfig(opts, )
    //TF_Session* sess = TF_NewSession(graph, options, status);
    //TF_DeleteSessionOptions(options);

    //C++ opts->config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.1);
    //C++ opts->config.mutable_gpu_options()->set_allow_growth(true);

    this->m_sess=TF_NewSession(graph, opts, status);
    TF_DeleteSessionOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << "load_nn: Error while creating new session, " <<status<< "\n";
    }
    
    TF_DeleteStatus(status);

}
TF_Tensor* NNEvaluator::make_input_tensor(const benzene::bitset_t &black_stones,
                                    const benzene::bitset_t &white_stones, benzene::HexColor toplay, 
                                    int boardsize) const {
    size_t i,j, k;
    int x,y;
    std::int64_t input_width=boardsize+2*m_input_padding;
    const std::vector<std::int64_t> input_dims = {1, input_width, input_width, m_input_depth};
    float input_vals[1][input_width][input_width][m_input_depth];
    //set the empty points
    for (i=0;i<input_width;i++){
        for(j=0;j<input_width;j++){
            for(k=0;k<m_input_depth; k++){
                //ten(0,i,j,k)=0;
                input_vals[0][i][j][k]=0;
                if(k==ToPlayEmptyPoints && i>=m_input_padding && j>=m_input_padding
                        && i<input_width-m_input_padding && j<input_width-m_input_padding){
                    //ten(0,i,j,k)=1.0;
                    input_vals[0][i][j][k]=1.0;
                    x=i-m_input_padding;
                    y=j-m_input_padding;
                    //static_assert(x>=0 && x<boardsize && y>=0 && y<boardsize, "error in make_input_tensor\n");
                    //be aware of this conversion! pari to int_move: x*boarsize + y
                    //ten_empty_points(0, x*boardsize+y)=true;
                    
                } 
                //toplay plane
                if(k==IndToPlay && toplay==benzene::WHITE){
                    //ten(0,i,j,k)=1.0;
                    input_vals[0][i][j][k]=1.0;
                }
            }
        }
    }

    //black stones, border padding
    for(i=0;i<m_input_padding;i++){
        for(j=0;j<input_width;j++){
            //ten(0,j,i,BlackStones)=1.0;
            input_vals[0][j][i][BlackStones]=1.0;
            //ten(0,j,input_width-1-i,BlackStones)=1.0;
            input_vals[0][j][input_width-1-i][BlackStones]=1.0;
        }
    }

    //white stones, border padding
    for(i=0;i<m_input_padding;i++){
        for(j=m_input_padding;j<input_width-m_input_padding;j++){
            //ten(0,i,j,WhiteStones)=1.0;
            input_vals[0][i][j][WhiteStones]=1.0;
            //ten(0,input_width-1-i,j,WhiteStones)=1.0;
            input_vals[0][input_width-1-i][j][WhiteStones]=1.0;
        }
    }

    //set m_board, and black/white played stones, modify empty points
    for (benzene::BitsetIterator it(black_stones); it; ++it){
        int p=*it-7;
        if (p<0) continue;
        benzene::HexPointUtil::pointToCoords(*it, x,y);
        //ten_empty_points(0, x*boardsize+y)=false;
        x += m_input_padding;
        y += m_input_padding;
        //m_board[x][y]=benzene::BLACK;
        //ten(0,x,y,BlackStones)=1.0;
        input_vals[0][x][y][BlackStones]=1.0;
        //ten(0,x,y,ToPlayEmptyPoints)=0.0;
        input_vals[0][x][y][ToPlayEmptyPoints]=0.0;
    }
    for (benzene::BitsetIterator it(white_stones); it; ++it){
        int p=*it-7;
        if(p<0) continue;
        benzene::HexPointUtil::pointToCoords(*it,x,y);
        //ten_empty_points(0, x*boardsize+y)=false;
        x += m_input_padding;
        y += m_input_padding;
        //m_board[x][y]=benzene::WHITE;
        //ten(0,x,y,WhiteStones)=1.0;
        input_vals[0][x][y][WhiteStones]=1.0;
        //ten(0,x,y,ToPlayEmptyPoints)=0.0;
        input_vals[0][x][y][ToPlayEmptyPoints]=0.0;
    }
    float *p_input_vals=(float*)input_vals;
    std::size_t data_size=1*input_width*input_width*m_input_depth*sizeof(float);
    TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT, input_dims.data(), input_dims.size(), p_input_vals, data_size);

    return input_tensor;
  //end
}

/*
 * return:
 * p: probability score of next moves
 * q: action-value for each next move
 * v: value estimate of current state
 */
float NNEvaluator::evaluate(const benzene::bitset_t &black, const benzene::bitset_t &white, benzene::HexColor toplay,
                            std::vector<float> &score, std::vector<float> & qValues, int boardsize) const {
    std::size_t input_width=boardsize+2*m_input_padding;
    auto t1=std::chrono::system_clock::now();
    TF_Tensor* x_input=make_input_tensor(black, white, toplay, boardsize);
    const std::vector<TF_Tensor*> input_tensors={x_input};

    std::string input_node="x_"+std::to_string(boardsize)+"x"+std::to_string(boardsize)+"_node";
    const std::vector<TF_Output> input_ops = {{TF_GraphOperationByName(this->m_graph, input_node.c_str()), 0}};

    std::string output_node="logits_"+std::to_string(boardsize)+"x"+std::to_string(boardsize)+"_node";
    std::string output_node_v=std::to_string(boardsize)+"x"+std::to_string(boardsize)+"_value";
    std::string output_node_q=std::to_string(boardsize)+"x"+std::to_string(boardsize)+"_q_values";
    const std::vector<TF_Output> out_ops = {{TF_GraphOperationByName(this->m_graph, output_node.c_str()), 0},
                                            {TF_GraphOperationByName(this->m_graph, output_node_v.c_str()),0},
                                            {TF_GraphOperationByName(this->m_graph, output_node_q.c_str()),0}};
    std::vector<TF_Tensor *> output_tensors={nullptr, nullptr, nullptr};
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(m_sess,
            nullptr, //Run options
            input_ops.data(), input_tensors.data(), 1, //input_op, input_tensor, num of input
            out_ops.data(), &output_tensors[0], output_tensors.size(), //output_op, output tensor, num of output
            nullptr, 0, //target operations, num of targets
            nullptr, //run metadata
            status
            );
    if(TF_GetCode(status)!=TF_OK){
        std::cout<<"error:"<<status<<"\n";
    }

    float* p_ret=static_cast<float*>(TF_TensorData(output_tensors[0]));
    float* v_ret=static_cast<float*>(TF_TensorData(output_tensors[1]));
    float* q_ret=static_cast<float*>(TF_TensorData(output_tensors[2]));
    float value_estimate=v_ret[0];

    /*
     * the score vector contains all logits for each move, whether valid or invalid
     * Note that indices of score should be converted into normal move
     * by x=i//boardsize, y=i%boardsize
     * where x is then converted into x+'a'
     * y -> y+1
     *
     * This conversion is different by what has been adopted by Benzene!
     */
    float max_value=-FLT_MAX;
    std::vector<int> empty_points;
    int max_ind=-1;
    float min_q_value=1.0;
    for(int i=0;i<boardsize*boardsize;i++){
        int x, y;
        x= i/boardsize;
        y= i%boardsize;
        size_t posi=benzene::HexPointUtil::coordsToPoint(x,y);
        //std::cout<<"i"<<i<<" "<<ret(i)<<" "<<q_ret(i)<<" "<<std::endl;
        //std::cout<<"i"<<i<<" "<<ret(i)<<" "<<std::endl;
        if(black.test(posi) || white.test(posi)){
            //ignore played points
            score[i]=0.0;
            continue;
        }
        score[i]=0.0;
        empty_points.push_back(i);
        if(max_value < p_ret[i]){
            max_value=p_ret[i];
            max_ind=i;
        }
        if(min_q_value>q_ret[i]){
            min_q_value=q_ret[i];
        }
        qValues[i]=(q_ret[i]+1.0)/2.0f;//convert to [0,1]
        score[i]=p_ret[i];
        //score[i]=(float)exp(score[i]);
      //  sum_value +=score[i];
    }
    double product_propagate_prob=1.0;
    float sum_value=0.0;
    for(int &i:empty_points){
        score[i]=(float)exp((score[i]-max_value));
        //score[i] +=max_value+1.0;
        sum_value = sum_value+score[i];
    }
    double avg_v=0.0;
    double q_value_with_max_p;
    double max_p=-1.0;
    for(int &i: empty_points){
        score[i]=score[i]/sum_value;
        avg_v += score[i]*(1.0-qValues[i]);
        score[i]=(1.0-m_q_weight_to_p)*score[i]+m_q_weight_to_p*(1.0-qValues[i]);
        if(score[i]>=0.01) 
            product_propagate_prob *= qValues[i];
        if(score[i]>max_p){
            max_p=score[i];
            q_value_with_max_p=(1.0-qValues[i]);
        }
    }
    //convert from [-1.0,1.0] to [0.0,1.0]
    double converted_min_q_value=(-min_q_value+1.0)/2.0f;
    converted_min_q_value=q_value_with_max_p;
    double converted_v_value=(1.0+value_estimate)/2.0f;
    value_estimate=(1.0-m_min_q_combine_weight)*converted_v_value+m_min_q_combine_weight*converted_min_q_value;
    value_estimate=(1.0-m_product_propagate_weight)*value_estimate + m_product_propagate_weight*(1.0-product_propagate_prob);

    //cleanup
    tf_utils::DeleteTensors(input_tensors);
    tf_utils::DeleteTensors(output_tensors);
    auto t2=std::chrono::system_clock::now();
    //std::cout<<"time cost per eva:"<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1e06<<" seconds\n";
    return value_estimate;
}
