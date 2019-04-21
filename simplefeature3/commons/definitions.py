import numpy as np

PADDINGS = 1

'''
input features description, 12 input planes in total

Feture plane index | Description
0 | Black stones
1 | White stones
2 | Empty points
3 | Toplay (0 for black, 1 for white)

---
so INPUT_DEPTH = 4
'''
INPUT_DEPTH = 4

class BuildInputTensor(object):
    def __init__(self, boardsize=9):
        self.boardsize=boardsize
        self._board = np.ndarray(dtype=np.int32, shape=(boardsize+2, self.boardsize+2))
        self.IndBlackStone = 0
        self.IndWhiteStone = 1
        self.IndEmptyPoint = 2
        self.IndToplay = 3


        self.NUMPADDING = 1

    def set_position_label_in_batch(self, batchLabels, kth, intNextMove):
        batchLabels[kth] = intNextMove

    '''A square board the same size as Tensor input, each point is either EMPTY, BLACK or WHITE
        used to check brige-related pattern,
        '''

    def _set_board(self, intMoveSeq):
        self._board.fill(HexColor.EMPTY)
        ''' set black padding boarders'''
        INPUT_WIDTH=self.boardsize+2
        for i in range(self.NUMPADDING):
            self._board[0:INPUT_WIDTH, i] = HexColor.BLACK
            self._board[0:INPUT_WIDTH, INPUT_WIDTH - 1 - i] = HexColor.BLACK
        ''' set white padding borders '''
        for j in range(self.NUMPADDING):
            self._board[j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
            self._board[INPUT_WIDTH - 1 - j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
        turn = HexColor.BLACK
        for intMove in intMoveSeq:
            x=intMove//self.boardsize
            y=intMove%self.boardsize
            #(x, y) = MoveConvertUtil.intMoveToPair(intMove)
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            self._board[x, y] = turn
            turn = HexColor.EMPTY - turn
            # B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11

    def makeTensorInBatch(self, batchPositionTensors, batchLabels, kth, intMoveSeq, intNextMove):
        self.set_position_label_in_batch(batchLabels, kth, intNextMove)
        self.set_position_tensors_in_batch(batchPositionTensors, kth, intMoveSeq)

    def makeTensorInPositionValueBatch(self, batchPositionTensors, batchValues, kth, intMoveSeq, value):
        batchValues[kth]=value
        self.set_position_tensors_in_batch(batchPositionTensors, kth, intMoveSeq)

    def set_position_tensors_in_batch(self, batch_positions, kth, intMoveSeq):
        INPUT_WIDTH=self.boardsize+2

        ''' set empty points first'''
        batch_positions[kth, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING,
        self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.IndEmptyPoint] = 1

        ''' set black occupied border  points'''
        for i in range(self.NUMPADDING):
            batch_positions[kth, 0:INPUT_WIDTH, i, self.IndBlackStone] = 1
            batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1 - i, self.IndBlackStone] = 1

        ''' set white occupied border points'''
        for j in range(self.NUMPADDING):
            batch_positions[kth, j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.IndWhiteStone] = 1
            batch_positions[kth, INPUT_WIDTH - 1 - j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING,
            self.IndWhiteStone] = 1

        self._set_board(intMoveSeq)
        turn = HexColor.BLACK
        ''' from filled square board, set black/white played stones and empty points in feature planes'''
        for intMove in intMoveSeq:
            #(x, y) = MoveConvertUtil.intMoveToPair(intMove)
            x=intMove//self.boardsize
            y=intMove%self.boardsize
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            ind = self.IndBlackStone if turn == HexColor.BLACK else self.IndWhiteStone
            batch_positions[kth, x, y, ind] = 1
            batch_positions[kth, x, y, self.IndEmptyPoint] = 0

            # set history plane
            # t +=1.0
            # batch_positions[kth,x,y, self.HISTORY_PLANE]=np.exp(-1.0/t)
            turn = HexColor.EMPTY - turn

        ''' set toplay plane, all one for Black to play, 0 for white'''
        if turn == HexColor.BLACK:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.IndToplay] = 0
        else:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.IndToplay] = 1


'''
Edge definitions

'''
NORTH_EDGE = -1
SOUTH_EDGE = -3
EAST_EDGE = -2
WEST_EDGE = -4

'''
HexColor definitions
'''


class HexColor:
    def __init__(self):
        pass

    BLACK, WHITE, EMPTY = range(1, 4)

if __name__ == "__main__":
    print("all definitions")
