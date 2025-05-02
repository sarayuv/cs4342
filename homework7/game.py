import numpy as np

#-------------------------------------------------------
class FrozenLake:
    '''A python implementation of the frozen lake game. 
        The list of actions for the player:
        ------------------------------
        0 : "LEFT"
        1 : "DOWN"
        2 : "RIGHT"
        3 : "UP"
        ------------------------------
        The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. 
        Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. 
        The agent is rewarded for finding a walkable path to a goal tile.
        The surface is described using a grid like the following:
        SFFF       (S: starting point, safe)
        FHFH       (F: frozen surface, safe)
        FFFH       (H: hole, fall to your doom)
        HFFG       (G: goal, where the frisbee is located)
        The game ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.
        For more details, please read https://gym.openai.com/envs/FrozenLake-v0/
'''
    # ----------------------------------------------
    def __init__(self, p=0.1, vector_state=False, image_state=False):
        ''' Initialize the game. 
            Inputs:
                p: probability of slippery move
            Outputs:
                self.n_s: the number of states of the machine, an integer scalar.
                self.row: the current row on the map, an integer scalar, initialized as 0.
                self.col: the current column on the map, an integer scalar, initialized as 0.
                self.map: the map of the game 
            Note: agent/player cannot access the above variables in the game. They are supposed to be hidden from the player.
        '''
        self.p = p 
        self.n_s = 16
        self.row = 0
        self.col = 0
        self.s= 0
        self.map=["SFFF",
                  "FHFH",
                  "FFFH",
                  "HFFG"]
        self.done = False # whether the game has ended yet
        self.vector_state = vector_state # whether or not to use a vector to represent a game state  (True: vector)
        self.image_state = image_state # whether returning the game state as an image (True) or a scalar (False)
        if image_state:
            self.map_image = np.zeros((1,9,9),dtype=np.float32)

    # ----------------------------------------------
    def game_state(self): # get game state
        if self.vector_state: # vector representation
            s=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
            s[self.s] = 1.
            return s
        elif self.image_state: # image representation
            x=self.map_image.copy()
            r,c = self.row,self.col # the current position of the agent
            x[0,2*r:2*r+2,2*c:2*c+2] = [[1,1],
                                        [1,1]]
            s = np.array(x,dtype=np.float32)
            return s
        else: # scalar representation
            return self.s

    # ----------------------------------------------
    def step(self, a):
        '''
           Given an action , return the reward and next state. 
           Input:
                a: the index of the lever being pulled by the agent. a is an integer scalar between 0 and n-1. 
                    n is the number of arms in the bandit.
           Output:
                s: the new state of the machine, an integer scalar. 
                r: the reward of the previous action, a float scalar. The "win" return 1., if "lose", return 0. as the reward.
                done: whether the game has ended, a boolean scalar. If the game has ended, return True. Otherwise, return False. 
        '''
        assert a in [0,1,2,3] # check if the action chosen by the player is valid
        assert not self.done # check if the game has already ended. if the game has ended, player can no longer move

        # if slippery, stay at the same location 
        slippery = np.random.choice([False,True], 1, p=[1.-self.p,self.p])

        if slippery:
            return self.game_state(), 0., False # slippery move (stay at the same location)

        if a==0:
            self.col=max(self.col-1,0)
        if a==1:
            self.row=min(self.row+1,3)
        if a==2:
            self.col=min(self.col+1,3)
        if a==3:
            self.row=max(self.row-1,0)
        self.s=self.row*4+self.col
        c = self.map[self.row][self.col]
        if c == 'H':
            self.done = True
            r = 0.
        elif c== 'G':
            self.done = True
            r = 1.
        else:
            r =0.
        return self.game_state(), r, self.done

    # ----------------------------------------------
    def run_games(self,player, N=1000):
        '''
            let the player play the game for N episodes. 
            Input:
                player: a player or agent that plays the frozen lake game. 
                N: the number of episodes that the player plays the game.
            Outputs:
                e: the average reward per game episode =  total sum of rewards collected / N, a float scalar. 
        '''
        Total_reward = 0. # initialize the total rewards
        # run N game episodes
        for _ in range(N):
            done=False
            s = self.game_state() # initial state
            # run 1 episode 
            while not done:
                # one game step
                a = player.choose_action(s) # player choose an action
                s_new,r,done = self.step(a) # play one game step 
                Total_reward+=r # add to the total rewards
                player.update_memory(s,a,s_new,r,done) # let player to update the statistics with the chosen action and received rewards.
                s = s_new
            # reset the game
            self.s=0
            self.row = 0
            self.col = 0
            self.done=False
        e = Total_reward / N  # compute the average reward per game
        return e




