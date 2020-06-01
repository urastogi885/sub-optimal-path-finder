def take_action(action, coordinates):
    """
    Call various actions based on an integer
    :param action: Varies from 0-7 to call one of the 8 defined actions
    :param coordinates: a tuple containing x-y coordinates of the node
    :return: new coordinates of the node after the desired action
    """

    if action == 0:
        return go_up(coordinates[0], coordinates[1])
    elif action == 1:
        return go_down(coordinates[0], coordinates[1])
    elif action == 2:
        return go_right(coordinates[0], coordinates[1])
    elif action == 3:
        return go_left(coordinates[0], coordinates[1])
    elif action == 4:
        return go_up_right(coordinates[0], coordinates[1])
    elif action == 5:
        return go_up_left(coordinates[0], coordinates[1])
    elif action == 6:
        return go_down_right(coordinates[0], coordinates[1])

    return go_down_left(coordinates[0], coordinates[1])


def go_up(x, y):
    """
    Go 1 unit in positive y-direction
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving a unit in the positive y-direction
    """
    return x, y + 1


def go_down(x, y):
    """
    Go 1 unit in negative y-direction
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving a unit in the negative y-direction
    """
    return x, y - 1


def go_right(x, y):
    """
    Go 1 unit in positive x-direction
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving a unit in the positive x-direction
    """
    return x + 1, y


def go_left(x, y):
    """
    Go 1 unit in negative x-direction
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving a unit in the negative x-direction
    """
    return x - 1, y


def go_up_right(x, y):
    """
    Go 1 unit in both positive x and y directions
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving diagonally up-right
    """
    return x + 1, y + 1


def go_up_left(x, y):
    """
    Go 1 unit in positive y-direction and 1 unit in negative x-direction
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving diagonally up-left
    """
    return x - 1, y + 1


def go_down_right(x, y):
    """
    Go 1 unit in negative y-direction and 1 unit in positive x-direction
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving diagonally down-right
    """
    return x - 1, y + 1


def go_down_left(x, y):
    """
    Go 1 unit in both negative x and y directions
    :param x: x-coordinate of the node
    :param y: y-coordinate of the node
    :return: new coordinates of the node after moving diagonally down-left
    """
    return x - 1, y - 1
