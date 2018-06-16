import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding

'''
TO-DO:
| Add render
'''
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
        self._wins = np.zeros([3,3], dtype=int) # to keep track of which local grids have been won
        self.observation_space = spaces.Box(np.array([-1,0,0,0], dtype=int), np.array([1,2,2,9], dtype=int), dtype=int)
        self.action_space = spaces.Box(np.array([0,0,-9], dtype=int), np.array([2,2,8], dtype=int), dtype=int)  # inclusive, hence 2 
        self.marker_map = {-1:color.BOLD+color.BLUE+'X'+color.END, 0:' ', 1:color.BOLD+color.GREEN+'O'+color.END}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] 

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
        
    
    def _checkWin(self):
        # if at all there is a win in a new local grid, it will happen in the one where the move was made
        marked_grid_idx = np.where(self.state[9,:,:].flatten() == 1)[0][0]
        grid = self.state[marked_grid_idx,:,:]
        # print(grid.shape)
        winner = self._checkLocalWin(grid.reshape(3,3))
        if winner == 0:
            return 0
        self._wins[int(marked_grid_idx/3), marked_grid_idx%3] = winner
        global_winner = self._checkLocalWin(self._wins)
        return global_winner
    
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
        winner = self._checkWin()
        done = (winner != 0)    # if winner is either -1 or 1, game is done
        r = 0
        return self.state, r, done, {'winner': winner}
    
    def render(self):
        if self.render_moves == False:
            pass
        for row in range(9):
            if row%3 == 0:
                print('\n')
            for col in range(9):
                if col%3 == 0:
                    print('\t', end='')
                print('['+self.marker_map[self.state[int(row/3)*3+int(col/3), row%3, col%3]]+']', end='')
            print('')
        print('\n')


        # print('\n')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\n')
        # print('\033[1m' + '\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]' + '\033[0m')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\n')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][ ][ ]')
        # print('\t[X][ ][ ]\t[ ][ ][ ]\t[ ][O][ ]')
        # print('\n')
        
    
    def set_render(self, visibility):
        self.render_moves = visibility