import random
import neuralnet as nn
import numpy as np

class Computer:
    """
    This class implements a computer player that plays intelligently.
    Rounds of the game are recorded. As the game progresses, a neural network
    is trained. The computer uses the trained neural network to make its
    selection.
    """
    def __init__(self):
        self.difficulty = 4 #number of players moves used to predict the next
        self.prob = nn.MultiClassifier(self.difficulty,15,5)
        
        self.player_choices = np.zeros(self.difficulty + 1, dtype = np.int64) #record players choices
        self.train_x = np.zeros((1, self.difficulty), dtype = np.int64) #players choices reshaped for training
        self.train_y = np.zeros(1, dtype = np.int64)
        
    def save_player_choice(self, player_choice, num_rounds):
        #saves players choices
        if num_rounds <= 5:
            self.player_choices[num_rounds - 1] = player_choice
        else:
            self.player_choices[0:self.difficulty] = self.player_choices[1:self.difficulty + 1]
            self.player_choices[self.difficulty] = player_choice

        #update the data for training           
        if num_rounds == 5:
            self.train_x[0,0:self.difficulty] = self.player_choices[0:self.difficulty]
            self.train_y[0] = self.player_choices[self.difficulty]
        elif num_rounds > 5:
            self.train_x = np.append(self.train_x, [self.player_choices[0:self.difficulty]], axis = 0)
            self.train_y = np.append(self.train_y, self.player_choices[self.difficulty])

        print(self.player_choices)
        print(self.train_x)
        print(self.train_y)

    def train(self):
        self.prob.train(self.train_x, self.train_y)

    def choice(self, num_rounds):
        """
        Returns a random integer between 0 and 4 for the first 10 rounds.
        After round 10 a neural network is trained based on the player's
        previous choices. The computer then predicts which choice the player
        is likely to make, and then randomly chooses a winning choice.
        """
        if num_rounds <= 10:
            computer_choice = random.randint(0,4)
        else:
            #Train
            self.train()
            player_choice = self.prob.predict([self.player_choices[1:]])
            print("Player is most likely to choose " + number_to_name(player_choice))

            #choose move for computer
            if player_choice == 0: #rock
                computer_choice = random.choice([1,2]) #rock loses to spock or paper
            elif player_choice == 1: #spock
                computer_choice = random.choice([2,3]) #spock loses to paper or lizard
            elif player_choice == 2: #paper
                computer_choice = random.choice([3,4]) #paper loses to lizard or scissors
            elif player_choice == 3: #lizard
                computer_choice = random.choice([0,4]) #lizard loses to rock or scissors
            else: #scissors
                computer_choice = random.choice([0,1]) #scissors loses to rock or spock
                
        return computer_choice

def name_to_number(name):
    """
    This functions returns the number associated with a name
    name = {"rock","spock","paper","lizard","scissors"}
    """
    if name == "rock":
        number = 0
    elif name == "spock":
        number = 1
    elif name == "paper":
        number = 2
    elif name == "lizard":
        number = 3
    else:
        #scissors
        number = 4

    return number

def number_to_name(number):
    """
    This functions returns the name associated with a number
    number = {0,1,2,3,4}
    """
    if number == 0:
        name = "rock"
    elif number == 1:
        name = "spock"
    elif number == 2:
        name = "paper"
    elif number == 3:
        name = "lizard"
    else:
        #4
        name = "scissors"

    return name

def get_player_choice():
    """
    Gets the players choice
    """
    while True:
        choice = input("""\
Please make a selection: 0 - rock, 1 - spock, 2 - paper, 3 - lizard, 4 - scissors
9 - quit
""")
        if choice in {"0","1","2","3","4","9"}:
            break
        
    return int(choice)

def winner(player_choice, computer_choice):
    """
    Determines the outcome of a game
    return value: 1 = player wins, 0 = tie, -1 = computer wins
    """
    diff = (player_choice - computer_choice) % 5
    if diff == 0:
        outcome = 0 #tie
    elif diff == 1 or diff == 2:
        outcome = 1 #player wins
    else:
        outcome = -1 #computer wins

    return outcome

def rpsls():
    """
    This functions plays a round of the game
    """

    wins = 0 # number of player wins
    loses = 0 # number of player loses
    ties = 0

    num_rounds = 0
    comp = Computer()

    while True:
        player_choice = get_player_choice()
        if player_choice == 9:
            break

        num_rounds = num_rounds + 1

        computer_choice = comp.choice(num_rounds)

        print("Computer chooses",number_to_name(computer_choice))
        print("Player chooses", number_to_name(player_choice))
        
        outcome = winner(player_choice, computer_choice)
        
        if outcome == 0:
            ties = ties + 1
            print("Tie!")
        elif outcome == 1:
            wins = wins + 1
            print("Player wins!")
        else:
            loses = loses + 1
            print("Computer wins!")

        #Save round
        comp.save_player_choice(player_choice, num_rounds)

        #update and print stats
        print(str(num_rounds) + " rounds: " + str(wins) + " wins / " + str(loses) + " loses / " + str(ties) + " ties")
       
rpsls()
