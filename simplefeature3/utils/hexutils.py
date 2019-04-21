
from commons.definitions import HexColor, NORTH_EDGE, SOUTH_EDGE, EAST_EDGE, WEST_EDGE

class GameCheck:
    def __init__(self):
        pass

    @staticmethod
    def winner(black_group, white_group):
        if (black_group.connected(NORTH_EDGE, SOUTH_EDGE)):
            return HexColor.BLACK
        elif (white_group.connected(WEST_EDGE, EAST_EDGE)):
            return HexColor.WHITE
        else:
            return HexColor.EMPTY

    @staticmethod
    def updateUF(intgamestate, black_group, white_group, intmove, player, boardsize=9):
        assert(player == HexColor.BLACK or player== HexColor.WHITE)
        x, y = intmove // boardsize, intmove % boardsize
        neighbors = []
        pattern = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1)]
        for p in pattern:
            x1, y1 = p[0] + x, p[1] + y
            if 0 <= x1 < boardsize and 0 <= y1 < boardsize:
                neighbors.append((x1, y1))
        if (player == HexColor.BLACK):
            if (y == 0):
                black_group.join(intmove, NORTH_EDGE)
            if (y == boardsize - 1):
                black_group.join(intmove, SOUTH_EDGE)

            for m in neighbors:
                m2 = m[0] * boardsize + m[1]
                if (m2 in intgamestate and list(intgamestate).index(m2) % 2 == player-1):
                    black_group.join(m2, intmove)
        else:

            if (x == 0):
                white_group.join(intmove, WEST_EDGE)
            if (x == boardsize - 1):
                white_group.join(intmove, EAST_EDGE)

            for m in neighbors:
                im = m[0] * boardsize + m[1]
                if (im in intgamestate and list(intgamestate).index(im) % 2 == player-1):
                    white_group.join(im, intmove)
        # print(black_group.parent)
        return (black_group, white_group)


def next_player(currentIntplayer):
    assert currentIntplayer == HexColor.BLACK or currentIntplayer== HexColor.WHITE
    return HexColor.EMPTY - currentIntplayer


def state_to_str(int_move_seq, boardsize):
    g=int_move_seq
    size=boardsize
    white = 'W'
    black = 'B'
    empty = '.'
    ret = '\n'
    coord_size = len(str(size))
    offset = 1
    ret+=' '*(offset+1)
    board=[None]*size
    for i in range(size):
        board[i]=[empty]*size

    for k, i in enumerate(g):
        x,y=i//size, i%size
        board[x][y]=black if k%2==0 else white

    PLAYERS = {"white": white, "black": black}
    for x in range(size):
        ret += chr(ord('A') + x) + ' ' * offset * 2
    ret += '\n'
    for y in range(size):
        ret += str(y + 1) + ' ' * (offset * 2 + coord_size - len(str(y + 1)))
        for x in range(size):
            if (board[x][y] == PLAYERS["white"]):
                ret += white
            elif (board[x][y] == PLAYERS["black"]):
                ret += black
            else:
                ret += empty
            ret += ' ' * offset * 2
        ret += white + "\n" + ' ' * offset * (y + 1)
    ret += ' ' * (offset * 2 + 1) + (black + ' ' * offset * 2) * size

    return ret


class MoveConvert:

    def __init__(self):
        pass

    @staticmethod
    def int_move_to_int_pair(int_move, boardsize):
        assert 1<=boardsize<=20
        x = int_move // boardsize
        y = int_move % boardsize
        return x, y

    @staticmethod
    def int_pair_to_int_move(pair, boardsize):
        x, y = pair
        return x * boardsize + y

    @staticmethod
    def raw_move_to_int_move(raw_move, boardsize):
        x = ord(raw_move[0].lower()) - ord('a')
        assert 0 <= x <= 25
        y = int(raw_move[1:]) - 1
        return x * boardsize + y

    @staticmethod
    def int_move_to_raw(int_move, boardsize):
        x, y = MoveConvert.int_move_to_int_pair(int_move, boardsize)
        y += 1
        return chr(x + ord('a')) + repr(y)

    @staticmethod
    def rotate_180(int_move, boardsize):
        assert (0 <= int_move < boardsize ** 2)
        return boardsize ** 2 - 1 - int_move
