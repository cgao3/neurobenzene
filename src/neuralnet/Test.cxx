#include "NeuroEvaluate.hpp"
#include<iostream>
#include<string>
#include<algorithm>
#include<iterator>
#include<sstream>

std::string toString(size_t p, int boardsize){
    size_t x,y;
    x=p/boardsize;
    y=p%boardsize;
    char a=(char)('a'+x);
    std::string res;
    res.push_back(a);
    res.append(std::to_string(y));
    return res;
}

int main(int argc, char *argv[]){
    if(argc<3){
        std::cout<<"Usage: ./nntest path_to_const_nn_model path_to_test_file\n";
        std::cout<<"Test file format: Each line is a sequence of black white moves + game result \n one line is one state-action-value tuple\n";
        return 1;
    }
    std::ifstream ifile(argv[2]);
    std::string line;

    std::string slpath=std::string(argv[1]);
    NNEvaluator nn(slpath);
    //To be finished this part!
    benzene::bitset_t b(std::string("0"));
    benzene::bitset_t w(std::string("0"));
    //std::cout<<"ABS_TOP_SRC_DIR: "<<ABS_TOP_SRCDIR <<"\n";
    std::cout<<"slmodelpath: "<<slpath<<"\n\n";
    std::cout<<b<<"\n";
    std::flush(std::cout);
    int boardsize=11;
    std::vector<float> score;
    score.resize(boardsize*boardsize);
    while(std::getline(ifile, line)){
        std::cout<<"processing:"<<line<<std::endl;
        std::istringstream iss(line);
        std::vector<string> tokens{std::istream_iterator<std::string>{iss},
                      std::istream_iterator<string>{}};
        b.reset();
        w.reset();
        int turn=0;
        for(std::string &token:tokens){
            //std::cout<<token<<std::endl;
            int len=token.length()-2-1;
            //std::cout<<token.substr(2,len)<<std::endl;
            std::string strmove=token.substr(2,len);
            benzene::HexPoint point=benzene::HexPointUtil::FromString(strmove);  
            int x=point;
            //std::cout<<"int:"<<x<<std::endl;
            //std::cout<<benzene::HexPointUtil::ToString(point)<<std::endl;
            if(turn%2==0){
                b[point]=1;
            } else{
                w[point]=1;
            }
            turn=(turn+1)%2;
        }
        if(turn==0){
            //nn.evaluate(b,w,benzene::BLACK,score, boardsize);
        } else{
            //nn.evaluate(b,w,benzene::WHITE,score, boardsize);
        }
    }
    b.reset();
    w.reset();
    b[7]=1;
    //nn.evaluate(b,w,benzene::WHITE,score, boardsize);
    std::cout<<"neural net output:";
    size_t best;
    std::cout<<"board: "<<benzene::HexPointUtil::ToString(b)<<"\n";
    std::cout<<"board: "<<benzene::HexPointUtil::ToString(w)<<"\n";
    float best_value=-0.1f;
    for(size_t i=0;i<score.size();i++){
        std::cout<<i<<"\t"<<toString(i,boardsize)<<":"<<score[i]<<"\n";
        if(best_value<score[i]){
            best=i;
            best_value=score[i];
        }
    }
    std::cout<<"\n";
    std::cout<<"best action is: "<<best<<"\t"<<toString(best,boardsize)<<"\n";
    return 0;
}
