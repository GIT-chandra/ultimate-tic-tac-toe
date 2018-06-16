from uttt_env import UltTTTEnv

class HumanPlayer():
    def __init__(self, player_num):
        self.player_num = player_num
        self.marker = 1
        if player_num == 2:
            self.marker = -1

    def play_move(self, local_grid=None):        
        print('\n Player', self.player_num, env.marker_map[self.marker])
        if local_grid == None:
            local_grid = int(input(' Choose local grid [0-8]: '))
        else:
            print(' Local grid is',local_grid,'based on previous move.')
        r = int(input(' Row: '))
        c = int(input(' Column: '))      
        if self.marker == -1:
            local_grid = -1*(local_grid + 1)  
        return [r, c, local_grid]


if __name__ == '__main__':
    env = UltTTTEnv()
    env.set_render(True)
    env.reset()

    print('\n\tWelcome to Ultimate Tic-Tac-Toe')
    print('\n\t\t 1. Human vs Human')

    choice = input('\n Choose an option : ')
    if choice == '1':        
        p1 = HumanPlayer(1)
        p2 = HumanPlayer(2)

    # Play the game
    done, count, local_grid = False, 0, None
    while done == False:
        action_valid = False
        while action_valid == False:
            if count%2 == 0:
                action = p1.play_move(local_grid)
            else:
                action = p2.play_move(local_grid)        
            # verify that action is valid
            local_grid_idx = action[2]
            if local_grid_idx < 0:
                local_grid_idx = abs(local_grid_idx) - 1
            if env.state[local_grid_idx, action[0], action[1]] == 0: # not yet marked
                action_valid = True
            else:
                env.render()
                print(' Selected cell is already occupied. Please choose another.')

        
        _, _, done, info = env.step(action)
        env.render()
        count += 1
        if env.wins[action[0], action[1]] != 0 or env.draws[action[0], action[1]] != 0:
            local_grid = None   # if the decided local grid has a win or draw, player is free to choose
        else: # restricting next local grid based on previous move
            local_grid = action[0]*3 + action[1]    

    if info['draw'] == 1:
        print('\t The game ended in a draw.') 
    elif info['winner'] == -1:
        print('\t Player 2 won!')
    else:
        print('\t Player 1 won!')

