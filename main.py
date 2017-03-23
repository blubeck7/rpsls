import random

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
    
def computer_choice():
    """
    Returns a random integer between 0 and 4
    """
    return random.randint(0,4)

def player_choice():
    """
    Gets the players choice
    """
    while True:
        choice = input("""\
Please make a selection: 0 - rock, 1 - spock, 2 - paper, 3 - lizard, 4 - spock
""")
        if choice in {"0","1","2","3","4"}:
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

    while True:
        player = player_choice()
        print("Player chooses",number_to_name(player))
        computer = computer_choice()
        print("Computer chooses",number_to_name(computer))
        outcome = winner(player, computer)
        if outcome == 0:
            print("Tie!")
        elif outcome == 1:
            print("Player wins!")
        else:
            print("Computer wins!")
    
        play = input("Play again? y/n:")
        if play != "y":
            break
        
rpsls()
