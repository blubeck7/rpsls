import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

class Computer:
    """
    This class implements a computer player that plays intelligently.
    Rounds of the game are recorded. As the game progresses, a neural network
    is trained. The computer uses the trained neural network to make its
    selection.
    """
    def __init__(self):
        self.difficulty = 3 #number of players moves computer remembers 
        
        self.rounds_p = [] #record players choices
        self.rounds_c = [] #record computer choices
        self.rounds_o = [] #record outcome of each round
        
        self.prob_rock = Sequential()
        self.prob_rock.add(Dense(5, activation='sigmoid', input_dim = 1))
        self.prob_win.add(Dense(1, activation='sigmoid'))
        self.prob_win.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.prob_tie = Sequential()
        self.prob_tie.add(Dense(5, activation='sigmoid', input_dim = 2))
        self.prob_tie.add(Dense(1, activation='sigmoid'))
        self.prob_tie.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.prob_loss = Sequential()
        self.prob_loss.add(Dense(5, activation='sigmoid', input_dim = 2))
        self.prob_loss.add(Dense(1, activation='sigmoid'))
        self.prob_loss.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
    def save_round(self, player_choice, outcome):
        self.rounds_c.append([player_choice, self.computer_choice])
        self.rounds_o.append(outcome)

    def print_rounds(self):
        for i in range(len(self.rounds_c)):
            print(self.rounds_c[i], self.rounds_o[i])

    def train(self):
        i = len(self.rounds_o)
        #printprob_win.layer
        if i > 2:
            print("Traing the probability of a win")
            self.prob_win.fit(self.rounds_c, self.rounds_o, batch_size = i)
            print("Traing the probability of a tie")
            self.prob_tie.fit(self.rounds_c, self.rounds_o, batch_size = i)
            print("Traing the probability of a loss")
            self.prob_loss.fit(self.rounds_c, self.rounds_o, batch_size = i)

    def choice(self, player_choice):
        """
        Returns a random integer between 0 and 4
        """
        i = len(self.rounds_o)
        if i > 2:
            p = np.array([player_choice])
            pw = self.prob_win.predict(p)
            pt = self.prob_tie.predict(p)
            pl = self.prob_loss.predict(p)
            print(pw)
        
        self.computer_choice = random.randint(0,4)
        
        return self.computer_choice

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

def player_choice():
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

    comp = Computer()

    while True:
        player = player_choice()
        print("Player chooses",number_to_name(player))
        if player == 9:
            break
        
        computer = comp.choice(player)
        print("Computer chooses",number_to_name(computer))
        outcome = winner(player, computer)
        
        if outcome == 0:
            print("Tie!")
        elif outcome == 1:
            print("Player wins!")
        else:
            print("Computer wins!")

        #Save round
        comp.save_round(player,outcome)
        comp.print_rounds()
        comp.train()
       
rpsls()
