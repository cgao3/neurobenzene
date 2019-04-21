import sys

class GTPInterface(object):
    def __init__(self, agent):
        self.agent = agent
        commands = {"name": self.gtp_name,
                    "genmove": self.gtp_genmove,
                    "quit": self.gtp_quit,
                    "showboard": self.gtp_show,
                    "play": self.gtp_play,
                    "list_commands": self.gtp_list_commands,
                    "clear_board": self.gtp_clear,
                    "boardsize": self.gtp_boardsize,
                    "close": self.gtp_close}

        self.commands = commands

    def send_command(self, command):
        p = command.split()
        func_key = p[0]
        args = p[1:]

        # ignore unknow commands
        if func_key not in self.commands:
            return True, "    "

        # call that function with parameters
        return self.commands[func_key](args)

    def gtp_name(self, args=None):
        return True, self.agent.name

    def gtp_list_commands(self, args=None):
        return True, self.commands.keys()

    def gtp_quit(self, args=None):
        if hasattr(self.agent, 'sess'):
            self.agent.sess.close()
        return True, "     "

    def gtp_clear(self, args=None):
        self.agent.reinitialize()
        return True, "    "

    def gtp_play(self, args):
        # play black/white a1
        assert (len(args) == 2)
        player=args[0].lower()
        if not (player[0]=='b' or player[0]=='w'):
            return False, 'Player should be black or white'
        raw_move=args[1].strip()
        assert ord('a') <= ord(raw_move[0]) <= ord('a')+self.agent.boardsize
        assert 1 <= int(raw_move[1:]) <= self.agent.boardsize

        self.agent.play_move(player, raw_move)
        return True, raw_move

    def gtp_genmove(self, args):
        assert (args[0][0] == 'b' or args[0][0] == 'w')
        player=args[0]
        raw_move = self.agent.generate_move(player)
        return True, raw_move

    def gtp_boardsize(self, args=None):
        boardsize=int(args[0])
        #print('boardsize: ', boardsize)
        assert (3<= boardsize <= 19)
        self.agent.set_boardsize(boardsize)
        return True, "    "

    def gtp_show(self, args=None):
        from utils.hexutils import state_to_str
        #print("agent boardsize, ", self.agent.boardsize)
        int_game_state=[]
        j1=0
        j2=0
        for i in range(len(self.agent.black_int_moves)+len(self.agent.white_int_moves)):
            if i%2==0:
                int_move=self.agent.black_int_moves[j1] 
                j1 += 1
            else:
                int_move=self.agent.white_int_moves[j2]
                j2 += 1
            int_game_state.append(int_move)
        if self.agent.verbose:
            print(state_to_str(int_game_state, self.agent.boardsize))
        return True, "        "

    def gtp_close(self, args=None):
        try:
            self.agent.sess.close()
        except AttributeError:
            pass
        return True, "    "
