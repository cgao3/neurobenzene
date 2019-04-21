#ifndef _NEURAL_H_
#define _NEURAL_H_
#include "Hex.hpp"
#include "Bitset.hpp"
#include "BitsetIterator.hpp"
#include <vector>
#include <tensorflow/c/c_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>

/*
 * Using a simple representation. 
 * Input to the neural net has 4 planes
 * 0: black stones
 * 1: white stones
 * 2: empty points
 * 3: toplay (all 0 for black, 1 for white)
 *
 * border padding is 1. 
 */

class NNEvaluator {

public:
    //Session *m_sess;
    TF_Session *m_sess;
    TF_Graph *m_graph;
    std::string m_neural_model_path;
    const static size_t m_input_padding=1;
    const static size_t m_input_depth=4;
    const static int BlackStones=0;
    const static int WhiteStones=1;
    const static int ToPlayEmptyPoints=2;
    const static int IndToPlay=3;
    double m_min_q_combine_weight;
    double m_q_weight_to_p;
    double m_product_propagate_weight;
public:
    NNEvaluator(std::string nn_model_path);
    NNEvaluator();
    ~NNEvaluator();

    void load_nn_model(std::string nn_model_path);
    float evaluate(const benzene::bitset_t &black, const benzene::bitset_t &white, benzene::HexColor toplay,
                   std::vector<float> &pScore, std::vector<float> &qValues, int boardsize) const;
    TF_Tensor* make_input_tensor(const benzene::bitset_t &black_stones, const benzene::bitset_t &white_stones,
            benzene::HexColor toplay, int boardsize) const;
};

#endif
