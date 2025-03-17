from time import sleep
import random
from tqdm import tqdm


def random_sampler(list):
    """
    Randomly sample an element from a list of elements
    :param list: List to sample from
    :return: Randomly sampled element
    """
    return random.choice(list)


def create_hyperparameter_combinations(nb_combinations, manual_combinations=None, nb_hidden_layers_= -1, hidden_layers_= -1, learning_rate_= -1.0 , batch_size_= -1, dropout_= -1.0, randomize=True, progressbar=True):
    """
    Create a list of different hyperparameter combinations to be used in the grid search
    :param nb_combinations: Maximum number of combinations to generate
    :param manual_combinations: List of manually generated combinations, added before the randomly generated combinations
    :param nb_hidden_layers_: fixed number of hidden layers (default: -1 for no fixed value, set a value if fixed)
    :param hidden_layers_: fixed hidden layers, provide a list with the size of each hidden layer, overwrites nb_hidden_layers_ (default: -1 for no fixed value, set a value if fixed)
    :param learning_rate_: fixed learning rate (default: -1.0 for no fixed value, set a value if fixed)
    :param batch_size_: fixed batch size (default: -1 for no fixed value, set a value if fixed)
    :param dropout_: fixed dropout (default: -1.0 for no fixed value, set a value if fixed)
    :param randomize: randomize (default: True) randomize the combinations (the manual combinations included)
    :param progressbar: show progress bar (default: True)
    :return: List of hyperparameter combinations (list of dictionaries {layer_sizes, learning_rate, batch_size, dropout})
    """

    hidden_layer_size_choices = [i for i in range(1000, 5000, 1000)]
    nb_hidden_layer_choices = [i for i in range(2, 4)]
    learning_rate_choices = [0.0001, 0.001, 0.01]
    batch_size_choices = [i for i in range(50, 250, 50)]
    dropout_choices = [0.0, 0.0, 0.1, 0.2]

    combinations = [] if manual_combinations is None else manual_combinations

    for f in tqdm(range(nb_combinations), desc='Generating combinations', disable=not progressbar):

        if f < len(combinations):
            continue

        nb_hidden_layers = random_sampler(nb_hidden_layer_choices) if nb_hidden_layers_ == -1 else nb_hidden_layers_

        if hidden_layers_ == -1:
            hidden_layers = []
            for j in range(nb_hidden_layers):
                hidden_layers.append(random_sampler(hidden_layer_size_choices))
        else:
            hidden_layers = hidden_layers_

        learning_rate = random_sampler(learning_rate_choices) if learning_rate_ == -1.0 else learning_rate_

        batch_size = random_sampler(batch_size_choices) if batch_size_ == -1 else batch_size_

        dropout = random_sampler(dropout_choices) if dropout_ == -1.0 else dropout_

        combination = {
            "layer_sizes": hidden_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout": dropout
        }

        if combination not in combinations:
            combinations.append(combination)

        sleep(0.1)

    if randomize:
        random.shuffle(combinations)

    return combinations


if __name__ == '__main__':
    print(create_hyperparameter_combinations(50))
