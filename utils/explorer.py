# Import necessary standard libraries
import cv2
import numpy as np
# Import custom-built methods
from utils import constants
from utils.actions import take_action


def check_node_validity(check_img, x, y):
    """
    Method to check whether point lies within any obstacle
    :param check_img: 2-d array with information of the map
    :param x: x-coordinate of the current node
    :param y: y-coordinate of the current node
    :return: false if point lies within any obstacle
    """
    # Check condition for out of bounds
    if x <= 0 or x >= constants.MAP_SIZE[1] or y <= 0 or y >= constants.MAP_SIZE[0]:
        return False
    # Check condition to verify if point lies within obstacle
    elif check_img[y, x].all() == 0:
        return False

    return True


class Explorer:
    def __init__(self, start_node, goal_node, method='d'):
        """
        Initialize the explorer with a start node and final goal node
        :param start_node: a tuple of starting x-y coordinates provided by the user
        :param goal_node: a tuple of goal coordinates provided by the user
        :param method: method to explore the map - 'd' for Dijkstra & 'a' for A-star
        """
        # Store puzzle and goal nodes as class
        self.start_node = start_node
        self.goal_node = goal_node
        self.method = method
        # Define an empty list to store all generated nodes
        self.generated_nodes = []
        # Define 2-D arrays to store information about generated nodes and parent nodes
        self.parent = np.full(fill_value=constants.NO_PARENT, shape=constants.MAP_SIZE)
        # Define video-writer of open-cv to record the exploration and final path
        video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.video_output = cv2.VideoWriter('exploration_' + self.method + '.avi', video_format, 200.0,
                                            (constants.MAP_SIZE[1], constants.MAP_SIZE[0]))

    def explore(self, map_img):
        """
        Method to explore the map to find the goal
        :param map_img: 2-d array with information of the map
        :return: nothing
        """
        # Initialize priority queue
        node_queue = []
        # Add start node in the array
        self.generated_nodes.append(self.start_node)
        self.parent[self.start_node[0]][self.start_node[1]] = constants.START_PARENT
        # Add start node to the queue
        node_queue.append(self.start_node)
        # Start exploring
        while len(node_queue):
            # Get node with minimum total cost
            current_node = node_queue[-1]
            # Add node to generated nodes array
            # Check for goal node
            if current_node == self.goal_node:
                break
            # Generate child nodes from current node
            for i in range(constants.MAX_ACTIONS):
                # Get coordinates of child node
                y, x = take_action(i, current_node)
                # Check whether child node is not within obstacle space and has not been previously generated
                if (check_node_validity(map_img, x, constants.MAP_SIZE[0] - y) and
                        self.parent[y][x] == constants.NO_PARENT):
                    # Add child node to priority queue
                    node_queue.append((y, x))
                    self.generated_nodes.append((y, x))
                    # Update parent of the child node
                    self.parent[y][x] = np.ravel_multi_index([current_node[0], current_node[1]],
                                                             dims=constants.MAP_SIZE)

    def generate_path(self):
        """
        Generate path using back-tracking
        :return: a list containing path nodes
        """
        # Define empty list to store path nodes
        # This list will be used to generate the node-path text file
        path_list = []
        # Get all data for goal node
        last_node = self.goal_node
        # Append the matrix for goal node
        path_list.append(last_node)
        # Iterate until we reach the initial node
        while self.parent[last_node[0]][last_node[1]] != constants.START_PARENT:
            # Search for parent node in the list of closed nodes
            last_node = np.unravel_index(self.parent[last_node[0]][last_node[1]], dims=constants.MAP_SIZE)
            path_list.append(last_node)
        # Return list containing all path nodes
        return path_list

    def show_exploration(self, map_img):
        """
        Show animation of map exploration and path from start to goal
        :param map_img: 2-d array with information of the map
        :return: nothing
        """
        blue = [255, 0, 0]
        white = [200, 200, 200]
        # Show all generated nodes
        for y, x in self.generated_nodes:
            map_img[constants.MAP_SIZE[0] - y, x] = white
            self.video_output.write(map_img)
        # Show path
        data = self.generate_path()
        for i in range(len(data) - 1, -1, -1):
            map_img[constants.MAP_SIZE[0] - data[i][0], data[i][1]] = blue
            self.video_output.write(map_img)
        # Resize image to make it bigger and show it for 15 seconds
        map_img = cv2.resize(map_img, (constants.MAP_SIZE[1] * 2, constants.MAP_SIZE[0] * 2))
        for _ in range(10000):
            self.video_output.write(map_img)
        self.video_output.release()
        cv2.destroyAllWindows()
