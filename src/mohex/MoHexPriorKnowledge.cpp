//----------------------------------------------------------------------------
/** @file MoHexPriorKnowledge.cpp */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "MoHexThreadState.hpp"
#include "MoHexPriorKnowledge.hpp"
#include "MoHexSearch.hpp"
#include <future>
#include <random>

using namespace benzene;

//----------------------------------------------------------------------------

MoHexPriorKnowledge::MoHexPriorKnowledge(const MoHexThreadState& state)
    : m_state(state)
{
}

MoHexPriorKnowledge::~MoHexPriorKnowledge()
{
}

SgUctValue MoHexPriorKnowledge::ProcessPosition(std::vector<SgUctMoveInfo>& moves,
                                          const HexPoint lastMove,
                                          const bool doPruning)
{
    if (m_state.Search().ProgressiveBiasConstant() == 0.0f)
        return 0.5;

    float dirichlet_noise=m_state.Search().RootDirichletPrior();
    float combine_weight=0.25;

 	bitset_t black_played_stones=m_state.State().Position().GetPlayed(BLACK) &
                                 m_state.State().Position().Const().GetCells();
    bitset_t white_played_stones=m_state.State().Position().GetPlayed(WHITE) &
                                 m_state.State().Position().Const().GetCells();

    int boardsize=m_state.State().Position().Width();
    std::vector<float> nn_score(boardsize*boardsize);
    std::vector<float> q_values(boardsize*boardsize);
    //m_state.Search().GetNNEvaluator().evaluate(black_played_stones, white_played_stones,m_state.State().ToPlay(), nn_score);

    auto evaFuture=std::async([this, &black_played_stones, &white_played_stones, &nn_score, &q_values, &boardsize]{
        return this->m_state.Search().GetNNEvaluator().evaluate(black_played_stones,
                                                         white_played_stones, this->m_state.State().ToPlay(), nn_score, q_values, boardsize);
    });

    bitset_t safe, pruned, consider;
    float totalGamma = 0.0f;
    float moveGamma[BITSETSIZE];
    const HexColor toPlay = m_state.ColorToPlay();
    const MoHexBoard& board = m_state.GetMoHexBoard();
    const MoHexPatterns& patterns = m_state.Search().GlobalPatterns();
    const MoHexPatterns& localPat = m_state.Search().LocalPatterns();
    for (std::size_t i = 0; i < moves.size(); )
    {
        const HexPoint move = (HexPoint)moves[i].m_move;
        const MoHexPatterns::Data* data;
        patterns.MatchWithKeysBoth(board.Keys(move), toPlay, &data);
        if (data == NULL)
        {
            consider.set(move);
            moveGamma[move] = 1.0f;
            totalGamma += 1.0f;
            ++i;
            continue;
        }
#if 0
        {
            uint64_t keys[3];
            MoHexPatterns::GetKeyFromBoard(keys, 12, board, move, toPlay);
            if ((keys[0] != board.Keys(move)[0])
                || (keys[1] != board.Keys(move)[1]))
            {
                LogInfo() << board.Write() << '\n'
                          << "move=" << move << '\n'
                          << " key[0]=" << keys[0] << '\n'
                          << "bkey[0]=" << board.Keys(move)[0] << '\n'
                          << " key[1]=" << keys[1] << '\n'
                          << "bkey[1]=" << board.Keys(move)[1] << '\n';
                throw BenzeneException("Keys don't match!");
            }
        }
#endif

        if (doPruning && data->type && !safe.test(move))
        {
            switch(data->type)
            {
            case 1:  // opponent filled
            case 2:  // vulnerable
                // Prune with no further checks
                pruned.set(move);
                std::swap(moves[i], moves.back());
                moves.pop_back();
                continue;
              
            case 3: // dominated
                { 
                    const HexPoint killerMove
                        = board.Const().PatternPoint(move, data->killer);
                    if (board.GetColor(killerMove) != EMPTY)
                    {
                        LogSevere() << board.Write() << '\n';
                        LogSevere() << "move=" << move << '\n';
                        LogSevere() << "killer=" << killerMove << ' ' 
                                    << data->killer << '\n';
                        LogSevere() << "gamma=" << data->gamma << '\n';
                        throw BenzeneException("Killer not empty!!!\n");
                    }
                    if (!pruned.test(killerMove))
                    {
                        safe.set(killerMove);
                        pruned.set(move);
                        if (moves.size() == 1)
                        {
                            LogSevere() << board.Write() << '\n';
                            LogSevere() << board.Write(pruned) << '\n';
                            LogSevere() << board.Write(safe) << '\n';
                            LogSevere() << "move=" << move << '\n';
                            LogSevere() << "killer=" << killerMove << '\n';
                            throw BenzeneException("Pruned dominated to empty!!\n");
                        }
                        std::swap(moves[i], moves.back());
                        moves.pop_back();
                        continue;
                    }
                }
            }
        }

        consider.set(move);
        moveGamma[move] = data->gamma;
        totalGamma += data->gamma;
        ++i;
    }
    if (lastMove != INVALID_POINT) 
    {
        for (int i = 1; i <= 12; ++i)
        {
            const HexPoint n = board.Const().PatternPoint(lastMove, i);
            if (consider.test(n))
            {
                const MoHexPatterns::Data* data;
                localPat.MatchWithKeysBoth(board.Keys(n), toPlay, &data);
                if (data != NULL) 
                {
                    moveGamma[n] += data->gamma;
                    totalGamma += data->gamma;
                }
            }            
        }
    }
	SgUctValue value_estimate=evaFuture.get();
    //m_state.SetNeuroValueEstimate(value_estimate);
    //std::cout<<"Get value estimate: "<<value_estimate<<" \n";
    //if (totalGamma < 1e-6)
    //    return value_estimate;

    typedef std::mt19937 G;
    typedef std::gamma_distribution<> D;
    G g(time(0));  // seed if you want with integral argument
    // http://en.wikipedia.org/wiki/Gamma_distribution
    double theta = 1;
    D d(dirichlet_noise+1e-8, theta);
    float cul=0.0;
    float arr[moves.size()];

    for(std::size_t i=0;i<moves.size();i++){
        float prob=d(g);
        cul +=prob;
        arr[i]=prob;
    }

    for (std::size_t i = 0; i < moves.size(); ++i)
    {
        const float gamma = moveGamma[moves[i].m_move];
        float prob;// = gamma / totalGamma;
        //changed by Chao Gao
        HexPoint sgMove=static_cast<HexPoint>(moves[i].m_move);
        int x,y;
        HexPointUtil::pointToCoords(sgMove, x, y);
        prob=nn_score[x*boardsize+y];
        if(dirichlet_noise>0.0)
            prob=0.25f*(arr[i]/cul)+0.75f*prob;
        //std::cout<<HexPointUtil::ToString(sgMove)<<":"<<prob<<"\t";
        moves[i].m_neuroActionValue=q_values[x*boardsize+y];
        moves[i].m_prior = prob;
        moves[i].m_gamma = gamma;
		moves[i].m_raveValue = 0.5f;
		moves[i].m_raveCount = 8;
    }
    return value_estimate;
}

//----------------------------------------------------------------------------
