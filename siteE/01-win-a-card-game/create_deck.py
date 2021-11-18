import matplotlib.pyplot as plt
from collections import Counter

def create_deck():
    """
    Create a list that represents the card deck

    Cards are represented by a number according to the following rules:
    - Cards from 2 to 10 are represented by their number
    - Jacks, Queens, and Kings (court cards) are represented by the number 10
    - Aces are represented by 11

    Card suits (clubs, diamonds, hearts, and spades) don't matter in the game, so they're not recorded.
    """
    numeric_cards = range(2,11)
    value_court_cards = 10
    n_court_cards = 3
    value_ace = 11

    cards_in_a_suit = list(numeric_cards) + [value_court_cards]*n_court_cards + [value_ace]
    deck = 4 * cards_in_a_suit

    return deck


def check_deck(deck):
    """
    Check if the counts per value are correct
    
    The asserts will raise an exception if there's any issue.
    If it's alright, it'll simply print the last message.
    """
    numeric_cards_except_ten = range(2,10)

    assert len(deck) == 52, 'The deck must have 52 cards'
    counts = Counter(deck)
    for val in numeric_cards_except_ten:
        assert counts[val] == 4, \
            'There should be 4 of each numeric card from 2 to 9 inclusive'

    assert counts[10] == 4*4, \
        'There should be 16 with value 10. The 10 itself + 3 courd cards for each of the 4 suits'

    assert counts[11] == 4, \
        'There should 4 aces, which are represented by 11'
    print('Deck is ok')
    
    
def plot_histogram(deck):
    """
    Plot a bar plot of counts of each card value
    
    Doing a standard bar plot instead of a histogram because the X axis' ticks look nicer this way
    """
    counts = Counter(deck)

    # doing this instead of getting .keys() and .values() separately to make sure they're in the same order
    x, y = list(zip(*counts.items()))

    plt.bar(x, y)
    plt.title('Count of cards')
    plt.ylabel('No. of cards')
    _=plt.xlabel('Card value')


def main():
    deck = create_deck()
    check_deck(deck)
    plot_histogram(deck)


if __name__ == '__main__':
    main()
