import random
import seaborn as sb
from itertools import product, chain

random.seed(0)


def create_deck():
    """
    Create a list that represents the card deck

    Cards are represented by a number according to the following rules:
    - Cards from 2 to 10 are represented by their number
    - Jacks, Queens, and Kings (court cards) are represented by the number 10
    - Aces are represented by 11

    Card suits (clubs, diamonds, hearts, and spades) don't matter in the game, so they're not recorded.
    
    Copied from create_deck.py
    """
    numeric_cards = range(2,11)
    value_court_cards = 10
    n_court_cards = 3
    value_ace = 11

    cards_in_a_suit = list(numeric_cards) + [value_court_cards]*n_court_cards + [value_ace]
    deck = 4 * cards_in_a_suit

    return deck


def alter_ace(hand):
    """
    Changes an ace from 11 to 1
    """
    hand.remove(11)
    hand.append(1)
    return hand


def simulate_one_game(deck, threshold):
    hand = [deck.pop(), deck.pop()]

    # there are exactly 2 aces
    # so use use of them as 1 instead of 11
    if sum(hand) == 22:
        hand = alter_ace(hand)

    while sum(hand) < threshold:
        hand.append(deck.pop())
        if sum(hand) > 21 and 11 in hand:
            hand = alter_ace(hand)
    
    return hand


def duel_play(threshold_1, threshold_2):
    """
    Simulate 2 strategies playing against each other
    
    Each strategy can have a different threshold for stopping.
    Cards are dealt first to one player until it finishes its game and then to the second.
    """
    deck = create_deck()
    random.shuffle(deck)
    
    sum_1 = sum(simulate_one_game(deck, threshold_1))
    sum_2 = sum(simulate_one_game(deck, threshold_2))

    winner = None

    if (sum_1 > 21 and sum_2 > 21) or sum_1 == sum_2:
        winner = 0
    elif sum_2 > 21:
        winner = threshold_1
    elif sum_1 > 21:
        winner = threshold_2
    # here I already know that both are smaller than 21 so I can check one against the other
    elif sum_1 > sum_2:
        winner = threshold_1
    elif sum_2 > sum_1:
        # or could be simply else I believe, but I'm being explicit
        winner = threshold_2

    return winner


def duel_all_combinations():
    """
    Duel all possible thresholds against each other
    
    Possible thresholds are from 10 to and including 19.
    """
    possible_thresholds = list(range(10, 20))
    all_possible_combinations = product(possible_thresholds, possible_thresholds)
    winners = [duel_play(threshold_1, threshold_2) for threshold_1, threshold_2 in all_possible_combinations]
    return winners


def run_simulation(n_simulations=100):
    """
    Run the simulations all vs all n times and plots a histogram of the winners
    """
    all_winners = list(chain(*(duel_all_combinations() for _ in range(n_simulations))))
    sb.histplot(all_winners, discrete=True)
    

if __name__ == '__main__':
    run_simulation(10000)