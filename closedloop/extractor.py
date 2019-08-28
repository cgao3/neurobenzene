from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import  re
import os
import argparse
import numpy as np
import sys

'''
Extract gamestates from a Hex game for learning
Hex game: a line of an alternating sequence of black and white moves + 1.0 (game result)
output data format: each line is a sequence of black white alternating moves plus value
'''
BLACK=1
WHITE=2
EMPTY=3
class Extractor(object):
    def __init__(self, src_file, dest_file, boardsize):
        self.src_file=open(src_file,"r")
        if dest_file == None:
            self.outwriter=sys.stdout
        else: 
            self.outwriter=open(dest_file,"w")
        self.boardsize=boardsize
        #self.parse_board_size()

    def parse_board_size(self):
        self.boardsize=0
        for line in self.src_file:
            if line.startswith("#"):
                array=line.split()
                for a in array:
                    if a.isdigit(): 
                        self.boardsize=int(a)
                        break
            if self.boardsize>0: break

        if self.boardsize==0: 
            print("boardsize not found in input file!")
            exit(1)

    def extract(self):
        self.outwriter.write("# "+repr(self.boardsize)+" x " + repr(self.boardsize) +" Hex state-action-value tuples\n")
        for line in self.src_file:
            if line.startswith("#"): 
                continue
            array=line.split()
            if not array[-1].replace('.','',1).replace('-','').isdigit(): continue
            res=float(array[-1])
            n=len(array) - 1
            for i in range(1, n):
                t=[]
                for move in array[0:i]:
                    idx=move.index(']')+1
                    t.append(move[0:idx])
                t.append(array[i])
                out_str=" ".join(t)
                relative_res=res if (i+1)%2==n%2 else -res
                self.outwriter.write(out_str)
                self.outwriter.write(" "+repr(relative_res)+"\n")
            #print(" ".join(array[0:i+1]))

    def parse_moveseq(self, line):
        moveseq=[]
        arr=line.split()
        for m in arr[0:-2]:
            j=m.index(']')+1
            moveseq.append(m[0:j])
        #moveseq.append(arr[-1])
        return moveseq, arr[-2], arr[-1]

    def postprocess(self, positionValuesFileName, just_shuffle=False):
        print("position-value postprocessing")
        boardsize=self.boardsize
        outfile=positionValuesFileName+"-post"
        if just_shuffle:
            cmd = "shuf "+positionValuesFileName+" >"+outfile
            os.system(cmd) 
            return 
        tenaryBoard=np.ndarray(shape=(boardsize*boardsize), dtype=np.int16)
        tt={}
        with open(positionValuesFileName) as f:
            for line in f:
                line=line.strip()
                if line.startswith("#"): continue
                movesequence, label, vstr=self.parse_moveseq(line)
                value = float(vstr)
                seq2=movesequence[:]
                seq2.append(label)
                j=label.index(']')+1
                movesequence.append(label[0:j])

                tenaryBoard.fill(0)
                turn=BLACK
                for m in movesequence:
                    m=m.strip()
                    move=m[2:-1]
                    x=ord(move[0].lower())-ord('a')
                    y=int(move[1:])-1

                    assert(0<=x<boardsize)
                    assert(0<=y<boardsize)
                    tenaryBoard[x*boardsize+y]=turn
                    turn = EMPTY - turn
                code=''.join(map(str,tenaryBoard))
                if code in tt:
                    mq, one_count, neg_one_count=tt[code]
                    if value>0.99:
                        one_count +=1
                    else:
                        neg_one_count +=1
                    tt[code]=(mq, one_count, neg_one_count)
                else:
                    one_count=0
                    neg_one_count=0
                    if value > 0.99:
                        one_count = 1
                    else:
                        neg_one_count = 1
                    tt[code]=(seq2, one_count, neg_one_count)

        print("size: ", len(tt))
        print("saved as", outfile) 
        with open(outfile, "w") as f:
            for line in tt.values():
                #print(line)
                mq, one_count, neg_one_count = line
                for m in mq:
                    f.write(m+' ')
                res=(one_count - neg_one_count )*1.0/(one_count+neg_one_count)
                f.write(repr(res)+'\n')

    def close(self):
        self.outwriter.close()
        self.src_file.close()
'''
shf format: 
    first line of the sgf file should contain the Hex board size, e.g.
    #13 x 13 Hex

    each line contains a sequence of b/w moves + 1.0 (one game) e.g.,
    B[a1] W[c2][a3 0.5; c2 0.5;] B[b3][b3 0.2; a6 0.8;] -1.0
    means B played a1, W played c2 with probability distribution from [..],
    B played b3 ...
    Note: The last value is w.r.t the player to play
output: 
    extract the game into gamestates, each line is a state e.g.,
    B[a1] W[c2][a3 0.5; a4 0.5;] 1.0
    B[a1] W[c2][a3 0.5; a4 0.5;] B[b3][b3 0.2; a6 0.8;] -1.0
'''
if __name__ == "__main__":
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", type=str, default=None, help='input shf games')
    parser.add_argument("--output_file", type=str, default=sys.stdout, help='output data file name')
    parser.add_argument("--boardsize", type=int, default=13, help='boardsize')
    parser.add_argument("--post_process", type=bool, default=True, help="whether do postprocessing, i.e., average the value of same game state")
    args=parser.parse_args()
    import sys
    if not args.input_file: 
        print("please indicate --input_file") 
        exit(1)
    if not args.output_file:
        print("output file should not be None")
        exit(1)
    ext=Extractor(args.input_file, args.output_file, boardsize=args.boardsize)
    ext.extract()
    ext.close()
    if args.post_process:
        ext.postprocess(args.output_file, just_shuffle=False)
    else:
        ext.postprocess(args.output_file, just_shuffle=True)
    exit(0)
