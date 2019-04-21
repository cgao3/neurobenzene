import numpy as np

PADDINGS = 1

'''
input features description, 12 input planes in total

Feture plane index | Description
0 | Black stones
1 | White stones
2 | Empty points
3 | Toplay (0 for black, 1 for white)
4 | Black bridge endpoints
5 | White bridge endpoints
6 | Toplay save bridge
7 | Toplay make connection
8 | Toplay form bridge
9 | Toplay block opponent's bridge
10 | Toplay block opponent's form bridge
11 | Toplay block opponent's make connection

---
so INPUT_DEPTH = 12

a more efficient algorithm to prepare a batch?

'''
INPUT_DEPTH = 12


class BuildInputTensor(object):
    def __init__(self, boardsize=9):
        self.boardsize=boardsize
        self._board = np.ndarray(dtype=np.int32, shape=(boardsize+2, self.boardsize+2))
        self.IndBlackStone = 0
        self.IndWhiteStone = 1
        self.IndEmptyPoint = 2
        self.IndToplay = 3
        self.IndBBridgeEndpoints = 4
        self.IndWBridgeEndpoints = 5
        self.IndToplaySaveBridge = 6
        self.IndToplayFormBridge = 7
        self.IndToplayMakeConnection = 8
        self.IndOppoSaveBridge = 9
        self.IndOppoFormBridge = 10
        self.IndOppoMakeConnection = 11

        self.NUMPADDING = 1

    def set_position_label_in_batch(self, batchLabels, kth, intNextMove):
        batchLabels[kth] = intNextMove

    '''A square board the same size as Tensor input, each point is either EMPTY, BLACK or WHITE
        used to check bridge-related pattern,
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
        return turn

    def makeTensorInBatch(self, batchPositionTensors, batchLabels, kth, intMoveSeq, intNextMove):
        self.set_position_label_in_batch(batchLabels, kth, intNextMove)
        self.set_position_tensors_in_batch(batchPositionTensors, kth, intMoveSeq)

    def get_bridge_neighbor(self, x, y, max_width):
        bridge_neighbor_list=[]
        nb_x, nb_y=x+1, y-2
        if nb_x < max_width and nb_y >=0:
            x1, y1 = x, y-1
            x2, y2 = x+1, y-1
            #x1,y1, x2, y2 are carriers
            bridge_neighbor_list.append((nb_x, nb_y, x1, y1, x2, y2))

        nb_x, nb_y = x + 2, y - 1
        if nb_x < max_width and nb_y >= 0:
            x1, y1 = x+1, y-1
            x2, y2 = x+1, y
            bridge_neighbor_list.append((nb_x, nb_y, x1, y1, x2, y2))

        nb_x, nb_y = x + 1, y + 1
        if nb_x < max_width and nb_y < max_width:
            x1, y1 = x+1, y
            x2, y2 = x, y+1
            bridge_neighbor_list.append((nb_x, nb_y, x1, y1, x2, y2))

        return bridge_neighbor_list

    def ge_bridge_neighbor_negative_direction(self, x, y, max_width):
        bridge_neighbor_list = []
        nb_x, nb_y = x - 1, y - 1
        if nb_x >=0 and nb_y >= 0:
            x1, y1 = x, y - 1
            x2, y2 = x - 1, y
            # x1,y1, x2, y2 are carriers
            bridge_neighbor_list.append((nb_x, nb_y, x1, y1, x2, y2))

        nb_x, nb_y = x - 2, y + 1
        if nb_x >=0 and nb_y < max_width:
            x1, y1 = x - 1, y
            x2, y2 = x - 1, y + 1
            bridge_neighbor_list.append((nb_x, nb_y, x1, y1, x2, y2))

        nb_x, nb_y = x - 1, y + 2
        if nb_x >=0 and nb_y < max_width:
            x1, y1 = x - 1, y + 1
            x2, y2 = x, y + 1
            bridge_neighbor_list.append((nb_x, nb_y, x1, y1, x2, y2))

        return bridge_neighbor_list

    def get_all_line_skip_neighbor(self, x, y, max_width):
        neighbor=[]
        nb_x, nb_y =  x-2, y
        x1,y1=x-1, y
        if nb_x >=0:
            neighbor.append((nb_x, nb_y, x1, y1))

        nb_x, nb_y = x, y-2
        x1,y1 = x, y-1
        if nb_y >=0:
            neighbor.append((nb_x, nb_y, x1, y1))

        nb_x, nb_y = x+2, y-2
        x1, y1 = x+1, y-1
        if nb_x< max_width and nb_y >=0:
            neighbor.append((nb_x, nb_y, x1, y1))

        nb_x, nb_y = x+2, y
        x1,y1=x+1, y1
        if nb_x < max_width:
            neighbor.append((nb_x, nb_y, x1, y1))
        nb_x, nb_y = x, y+2
        x1,y1=x,y+1
        if nb_y < max_width:
            neighbor.append((nb_x, nb_y, x1, y1))

        nb_x, nb_y = x-2, y+2
        x1,y1=x-1,y+1
        if nb_x >=0 and nb_y< max_width:
            if nb_y < max_width:
                neighbor.append((nb_x, nb_y, x1, y1))
        return neighbor

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

        toplay=self._set_board(intMoveSeq)

        ''' set toplay plane, all one for Black to play, 0 for white'''
        if toplay == HexColor.BLACK:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.IndToplay] = 0
        else:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.IndToplay] = 1

        '''
        Layout of the board
        (i,j)   -- (i+1,j)
          |     /    |
        (i,j+1) --(i+1,j+1)
        '''
        turn = HexColor.BLACK
        ''' from filled square board, set black/white played stones, empty points and other features
        in feature planes'''

        for intMove in intMoveSeq:
            x=intMove//self.boardsize
            y=intMove%self.boardsize
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            ind = self.IndBlackStone if turn == HexColor.BLACK else self.IndWhiteStone
            batch_positions[kth, x, y, ind] = 1
            batch_positions[kth, x, y, self.IndEmptyPoint] = 0
            #set black/white bridge endpoints, toplay save/form bridge, make connection, opponent saver/form bridge, make connection
            for nb_x, nb_y, x1, y1, x2, y2 in self.get_bridge_neighbor(x,y, max_width=INPUT_WIDTH):
                if self._board[nb_x, nb_y]== turn and self._board[x1, y1]!=HexColor.EMPTY-turn and self._board[x2, y2]!=HexColor.EMPTY-turn:
                    #bridge endpoints
                    ind = self.IndBBridgeEndpoints if turn == HexColor.BLACK else self.IndWBridgeEndpoints
                    batch_positions[kth, nb_x, nb_y, ind] = 1
                    batch_positions[kth, x, y, ind] = 1

                #save bridge
                if self._board[nb_x, nb_y] == turn and self._board[x1, y1]==HexColor.EMPTY and self._board[x2, y2]==HexColor.EMPTY-turn:
                    ind = self.IndToplaySaveBridge if turn == toplay else self.IndOppoSaveBridge
                    batch_positions[kth, x1,y1, ind] = 1
                if self._board[nb_x, nb_y] == turn and self._board[x1, y1]==HexColor.EMPTY-turn and self._board[x2, y2] == HexColor.EMPTY:
                    ind = self.IndToplaySaveBridge if turn == toplay else self.IndOppoSaveBridge
                    batch_positions[kth, x2, y2, ind]=1

                #form bridge
                if self._board[x1,y1]==HexColor.EMPTY and self._board[x2,y2]==HexColor.EMPTY:
                    if self._board[nb_x, nb_y]==HexColor.EMPTY:
                        ind = self.IndToplayFormBridge if turn == toplay else self.IndOppoFormBridge
                        batch_positions[kth, nb_x, nb_y, ind]=1

            for nb_x, nb_y, x1, y1, x2, y2 in self.ge_bridge_neighbor_negative_direction(x,y, max_width=INPUT_WIDTH):
                #form bridge, negaive direction
                if self._board[x1, y1] == HexColor.EMPTY and self._board[x2, y2] == HexColor.EMPTY:
                    if self._board[nb_x, nb_y] == HexColor.EMPTY:
                        ind = self.IndToplayFormBridge if turn == toplay else self.IndOppoFormBridge
                        batch_positions[kth, nb_x, nb_y, ind] = 1

            for nb_x, nb_y, x1, y1 in self.get_all_line_skip_neighbor(x,y, max_width=INPUT_WIDTH):
                if self._board[nb_x, nb_y] == turn and self._board[x1,y1]==HexColor.EMPTY:
                    ind = self.IndToplayMakeConnection if turn ==toplay else self.IndOppoMakeConnection
                    batch_positions[kth, x1, y1, ind]=1

            turn = HexColor.EMPTY - turn


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
