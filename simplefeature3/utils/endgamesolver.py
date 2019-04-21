from play.tournaments.program import Program
import sys
import argparse
from play.tournaments.wrapper import Wrapper
import os
import random

def gen_random_positions_and_solve(boardsize, player, num_positions):
    outfile=open('rand'+repr(boardsize)+'x'+repr(boardsize)+'.txt','a')
    states=[]
    for i in range(num_positions):
        n_moves=random.randint(11, boardsize*boardsize/2)
        state=[]
        moves_list=range(boardsize*boardsize)
        toplay=0
        for j in range(n_moves):
            m=random.choice(moves_list)
            x,y=m//boardsize + ord('a'), m%boardsize + 1
            xchar=chr(x)
            if toplay==0:
                str_move='B['+xchar+repr(y)+']'
            else:
                str_move='W['+xchar+repr(y)+']'
            toplay=(toplay+1)%2
            state.append(str_move)
            moves_list.remove(m)
        states.append(state)
    cnt=0
    for s in states:
        #player.reconnect()
        player.sendCommand('clear_board')
        player.sendCommand('dfpn-clear-tt')
        player.sendCommand('boardsize '+repr(args.boardsize))
        toplay=0
        for m in s:
            i_start=m.index('[')+1
            i_end=m.index(']')
            m=m[i_start:i_end]
            if toplay ==0: 
                player.sendCommand('play b '+m)
            else:
                player.sendCommand('play w '+m)
            toplay=(toplay+1)%2
        if toplay==0: 
            ret=player.sendCommand('dfpn-solve-state b')
        else:
            ret=player.sendCommand('dfpn-solve-state w')
        print(cnt,'ret=',ret)
        cnt +=1
        line=' '.join(s)
        outfile.write(line+' ')
        if ret.find('black')>=0:
            if toplay==0:
                outfile.write('1.0\n')
            else:
                outfile.write('-1.0\n')
        else:
            if toplay==0:
                outfile.write('-1.0\n')
            else:
                outfile.write('1.0\n')
        outfile.flush()
    outfile.close()
    player.sendCommand('quit')
    

if __name__ == "__main__":
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--endgamefile', type=str, default='', help='path to endgame file')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument("--with_value", type=bool, default=True, help="endgamedata with value?")
    parser.add_argument('--exe_path', type=str, default='', help='path to executable solver (MoHex)')
    parser.add_argument('--boardsize', type=int, default=8, help='board size to play on')
    parser.add_argument("--verbose", help="verbose?", action="store_true", default=False)
    parser.add_argument('--gen_rand_solve', action='store_true', default=False, help='generate random positions and solve') 
    args=parser.parse_args()

    player=Wrapper(args.exe_path, args.verbose)
    if args.gen_rand_solve:
        print('randomly generating positions\n')
        gen_random_positions_and_solve(args.boardsize, player, 100000)
        sys.exit(0)

    if not os.path.isfile(args.endgamefile):
        print('file does not exist, use --help')
        sys.exit(-1)

    states=[]
    if os.path.isfile(args.endgamefile):
        with open(args.endgamefile, 'r') as f:
            for line in f:
                s=line.strip().split()
                if line.strip().startswith('#'): continue
                if args.with_value:
                    s=s[0:-1]
                states.append(s)
    
    outfile=open(args.output,'w')
    cnt=0
    for s in states:
        player.reconnect()
        player.sendCommand('dfpn-clear-tt')
        player.sendCommand('clear_board')
        player.sendCommand('boardsize '+repr(args.boardsize))
        player.sendCommand('param_dfpn timelimit 2')
        toplay=0
        for m in s:
            i_start=m.index('[')+1
            i_end=m.index(']')
            m=m[i_start:i_end]
            if toplay ==0: 
                player.sendCommand('play b '+m)
            else:
                player.sendCommand('play w '+m)
            toplay=(toplay+1)%2
        if toplay==0: 
            ret=player.sendCommand('dfpn-solve-state b')
        else:
            ret=player.sendCommand('dfpn-solve-state w')
        print(cnt,'ret=',ret)
        cnt +=1
        line=' '.join(s)
        if ret.find('black')>=0:
            outfile.write(line+' ')
            if toplay==0:
                outfile.write('1.0\n')
            else:
                outfile.write('-1.0\n')
        elif ret.find('white')>=0:
            outfile.write(line+' ')
            if toplay==0:
                outfile.write('-1.0\n')
            else:
                outfile.write('1.0\n')
        else: continue
        outfile.flush()
    outfile.close()
    player.sendCommand('quit')
