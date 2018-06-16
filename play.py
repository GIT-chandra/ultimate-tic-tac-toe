from uttt_env import UltTTTEnv

class HumanPlayer():
    def __init__(self, player_num):
        self.player_num = player_num
        self.marker = 1
        if player_num == 2:
            self.marker = -1

    def play_move(self):
        print(' Player', self.player_num)
        print(' Choose local grid [0-8]]: ')
        local_grid = int(input())
        print(' Row: ')
        r = int(input())
        print(' Column: ')
        c = int(input())      
        if self.marker == -1:
            local_grid = -1*(local_grid + 1)  
        return [r, c, local_grid]


if __name__ == '__main__':
    env = UltTTTEnv()
    env.set_render(True)
    env.reset()

    print('\n\tWelcome to Ultimate Tic-Tac-Toe. Choose an option')
    print('\n\t\t 1. Human vs Human')

    choice = input()
    if choice == '1':
        done, count = False, 0
        p1 = HumanPlayer(1)
        p2 = HumanPlayer(2)
        while done == False:
            if count%2 == 0:
                _, _, done, info = env.step(p1.play_move())
            else:
                _, _, done, info = env.step(p2.play_move())
            env.render()
            count += 1
            
        if info['winner'] == -1:
            print('\t Player 2 won!')
        else:
            print('\t Player 1 won!')

