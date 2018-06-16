import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class UltTTTEnv(gym.Env):
    def __init__(self):    
        self.render_moves = False
        self.wins = np.zeros([3,3], dtype=int) # to keep track of which local grids have been won
        self.draws = np.zeros([3,3], dtype=int)
        self.observation_space = spaces.Box(np.array([-1,0,0,0], dtype=int), np.array([1,2,2,9], dtype=int), dtype=int)
        self.action_space = spaces.Box(np.array([0,0,-9], dtype=int), np.array([2,2,8], dtype=int), dtype=int)  # inclusive, hence 2 
        self.marker_map = {-1:color.BOLD+color.BLUE+'X'+color.END, 0:' ', 1:color.BOLD+color.GREEN+'O'+color.END}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] 

    ''' Check for win before checking draw! '''

    def _checkLocalWin(self, grid):
        sums = list(np.sum(grid, axis = 0))
        sums.extend(list(np.sum(grid, axis = 1)))
        sums.append(grid[0,0] + grid[1,1] + grid[2,2])  # diagonal
        sums.append(grid[0,2] + grid[1,1] + grid[2,0])  # other diagonal
        if max(sums)>2:
            return 1
        elif min(sums)<-2:
            return -1
        return 0
    
    def _checkLocalDraw(self, grid):
        if np.where(grid.flatten() == 0)[0].shape[0] == 0:
            return 1
        return 0        
    
    def _checkWinDraw(self, marked_grid_idx):
        draw = 0
        grid = self.state[marked_grid_idx,:,:].reshape(3,3)
        winner = self._checkLocalWin(grid)
        if winner == 0: # check for a draw
            draw = self._checkLocalDraw(grid)
            if draw == 0:    # then, surely the game is not a draw
                return draw, 0
            # else, update the draw data and check if the whole game ends in draw
            if self.wins[int(marked_grid_idx/3), marked_grid_idx%3] == 0:   # only grids not previously won will be considered
                self.draws[int(marked_grid_idx/3), marked_grid_idx%3] = draw
            # will be a draw when all local grids are EITHER won or in draw
            global_draw = self._checkLocalDraw(self.draws + self.wins)    
            return global_draw, 0

        # update the wins data and check if a player has won the game
        if self.wins[int(marked_grid_idx/3), marked_grid_idx%3] == 0:   # only grids not previously won will be considered
            self.wins[int(marked_grid_idx/3), marked_grid_idx%3] = winner        
        global_winner = self._checkLocalWin(self.wins)
        return draw, global_winner
    
    def reset(self):        
        self.state = np.zeros([10,3,3], dtype=int)
        return self.state        

    def step(self, action):    
        marker = 1
        local_grid_idx = action[2]
        if local_grid_idx < 0:
            local_grid_idx = abs(local_grid_idx) - 1
            marker = -1
        self.state[local_grid_idx, action[0], action[1]] = marker
        self.state[9,:,:] *= 0
        self.state[9,action[0], action[1]] = 1
        # if at all there is a win in a new local grid, it will happen in the one where the move was made
        draw, winner = self._checkWinDraw(local_grid_idx)
        done = (winner != 0 or draw)    # if winner is either -1 or 1, game is done
        r = 0
        return self.state, r, done, {'winner': winner, 'draw': draw}
    
    def render(self):
        if self.render_moves == False:
            pass
        for row in range(9):
            if row%3 == 0:
                print('\n')
            for col in range(9):
                if col%3 == 0:
                    print('\t', end='')
                if self.state[9, int(row/3), int(col/3)] == 1:  # selected local grid; highlight
                    print(color.RED + '[' + color.END, end='')
                    print(self.marker_map[self.state[int(row/3)*3+int(col/3), row%3, col%3]], end='')
                    print(color.RED + ']' + color.END, end='')
                    # broken down into three because self.marker_map already has color.END
                else:
                    print('['+self.marker_map[self.state[int(row/3)*3+int(col/3), row%3, col%3]]+']', end='')
            print('')
        print('\n -- Global scores --')
        for row in range(3):
            for col in range(3):
                if self.draws[row, col] == 1:
                    print('[*]', end='')
                else:
                    print('['+self.marker_map[self.wins[row, col]]+']', end='')
            print('')

        
    
    def set_render(self, visibility):
        self.render_moves = visibility