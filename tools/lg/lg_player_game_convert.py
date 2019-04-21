#!/usr/bin/env python3
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import re

"""
convert little player's game into simple hex game format
simple hex format represents hex game as sequence of alternating moves plus game result 1 or -1
result is with respect to the player to play in the last step

Example game in little-golem: 
    (;FF[4]EV[hex.ch.23.1.1]PB[Daniel Sepczuk]PW[Maciej Celuch]SZ[13]RE[W]GC[ game #1139802]SO[http://www.littlegolem.com];W[mf];B[swap];W[kj];B[ie];W[fh];B[ed];W[cd];B[cb];W[dd];B[fb];W[db];B[ii];W[ef];B[dj];W[ld];B[kd];W[hi];B[hl];W[ej];B[dk];W[el];B[fg];W[dg];B[dh];W[eh];B[eg];W[kf];B[ke];W[le];B[lc];W[gf];B[gg];W[hf];B[ig];W[hg];B[gh];W[hh];B[fk];W[resign])

Explain: 
    FF: game type, here Hex
    PB: black player's name
    PW: white player's name
    SZ: boardsize
    RE: who wins
    B[mf]: B played at m6
    W[swap]: W swaps

How to convert a game to normal game when there is `swap`?
 --- the move before `swap` remains unchanged. Here mf => mf, equivalently m6
 --- exchange two character of moves after `swap`, kj => jk, equivalently j11
 The result will be an equivalent Hex game without `swap`.

NOTE: in lg game, W starts first, 
but RE[B] means first player win, RE[W] means second player win
"""

pattern_result=r'RE\[[B|W]\]'
pattern_move=r';[B|W]\[[a-z][a-z]\]'
pattern_swap=r';[B|W]\[swap\]'
pattern_13=r'SZ\[13\]'
pattern_11=r'SZ\[11\]'
pattern_19=r'SZ\[19\]'

BLACK=1
WHITE=2
EMPTY=3
class LittleGolem(object):
    BLACK="B"
    WHITE="W"
    EMPTY=None
    def __init__(self, input_lg_file, output_file):
        """
        param input_lg_file: text file, each contains one lg game
        param dest_file: text file, output file, each line contains one converted game
        """
        self.input_file=input_lg_file
        self.output_file_prefix=output_file
        self.least_num_moves=5

    def convert(self):
        output_11=self.output_file_prefix+repr(11)+'x'+repr(11)+'.shf'
        output_13=self.output_file_prefix+repr(13)+'x'+repr(13)+'.shf'
        output_19=self.output_file_prefix+repr(19)+'x'+repr(19)+'.shf'
        file_out_11=open(output_11, 'w')
        file_out_13=open(output_13, 'w')
        file_out_19=open(output_19, 'w')
        cnt=0
        with open(self.input_file, "r") as file_in:
            for lg_game in file_in:
                lg_game = lg_game.strip()
                if len(lg_game) == 0:
                    continue
                sz=self.get_boardsize(lg_game)
                if sz !=11 and sz!=13 and sz!=19:
                    # unrecongized board size, little golem supports only 11x11,13x13,19x19
                    continue
                fout=file_out_11
                swap=self.has_swap(lg_game)
                if sz == 13:
                    fout=file_out_13
                elif sz == 19:
                    fout=file_out_19
                winner=self.get_game_result(lg_game)
                if winner == EMPTY:
                    # ignore game if result is unknown
                    continue
                moves=self.get_all_moves(lg_game, swap)
                moves=self.moves_post_process(moves)
                if len(moves) <= self.least_num_moves:
                    # ignore too short games
                    continue
                game=' '.join(moves)
                game +=' '
                if swap:
                    # if swap, RE[B] means second player win
                    # RE[W] means first player win.
                    # So invert to normal 
                    winner = self._invert_result(winner)
                N=len(moves)
                last_to_play=BLACK if N%2==0 else WHITE
                if last_to_play == winner:
                    game +='1'
                else: game +='-1'
                fout.write(game+'\n')
                cnt +=1
        file_out_11.close()
        file_out_13.close()
        file_out_19.close()
    
    def moves_post_process(self,moves):
        toplay=0
        moves2=[]
        for m in moves:
            c='B' if toplay==0 else 'W'
            moves2.append(c+'['+m+']')
            toplay = (toplay+1)%2
        return moves2
    def _invert_result(self, result):
        assert(result in [BLACK, WHITE])
        if result == BLACK:
            return WHITE
        return BLACK

    def convert_move(self, lg_move):
        """
        convert a little-golem move to 
        noraml move
        mf => m6
        """
        x=lg_move[0].lower()
        y=ord(lg_move[1].lower()) - ord('a') + 1
        return x+repr(y)

    def get_all_moves(self, lg_game, swap=False):
        lg_moves=re.findall(pattern_move, lg_game)
        moves=[]
        for k, m in enumerate(lg_moves):
            m=m[3:-1] # ;B[mf] => mf
            assert(len(m) == 2)
            if swap and k>=1:
                m=m[1]+m[0] # reverse
            moves.append(self.convert_move(m))
        return moves
    
    def get_game_result(self, lg_game):
        """
        find the winner from text lg_game 
        """
        ret=re.findall(pattern_result, lg_game)
        if len(ret)<=0:
            return EMPTY
        if len(ret) > 1:
            print('ERROR! Multiple lg game results exist', ret)
            return EMPTY
        ret=ret[0]
        if 'B' in ret:
            return BLACK
        elif 'W' in ret:
            return WHITE
        else:
            return EMPTY

    def has_swap(self, lg_game):
        ret=re.findall(pattern_swap, lg_game)
        if len(ret)<=0:
            return False
        return True
    
    def get_boardsize(self, lg_game):
        matches11=re.findall(pattern_11, lg_game)
        matches13=re.findall(pattern_13, lg_game)
        matches19=re.findall(pattern_19, lg_game)
        if len(matches11)>0:
            return 11
        if len(matches13)>0:
            return 13
        if len(matches19)>0:
            return 19
        return 'unknown_size'

if __name__ =="__main__":
    import argparse
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_file', type=str, default='', help='path to player game file')
    parser.add_argument('--output_file', type=str, default='output', help='path to output file')
    args=parser.parse_args()
    if args.input_file=='':
        print("usage ./program --input_file=players_games.txt --output_file=output_file.txt")
        import sys
        sys.exit(1)
    lg=LittleGolem(args.input_file, args.output_file)
    lg.convert()
    print('Done.\nSaved to '+args.output_file+'[11,13,19]x[11,13,19].shf')
