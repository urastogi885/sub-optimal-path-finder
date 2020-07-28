# Import necessary standard libraries
import ast
from sys import argv
from time import time
# Import necessary custom-built classes and methods
from utils.obstacle_space import Map
from utils.explorer import Explorer, check_node_validity


"""
Add various parameters as input arguments from user
:param start_node_data: a tuple of 2 values: start coordinates
:param goal_node_data: a tuple of 2 values: goal coordinates
:param robot_radius: radius of the robot
:param clearance: minimum distance between robot and any obstacle 
:param method: 0 for weighted a-star and 1 for depth-first search; look at DICT_METHODS in constants.py for more info
:param animation: 1 to show animation otherwise use 0
"""
script, start_node_coords, goal_node_coords, robot_radius, clearance, method, animation = argv

if __name__ == '__main__':
    # Convert arguments into their required data types and formats
    start_node_coords = tuple(ast.literal_eval(start_node_coords))
    start_node_coords = start_node_coords[1], start_node_coords[0]
    goal_node_coords = tuple(ast.literal_eval(goal_node_coords))
    goal_node_coords = goal_node_coords[1], goal_node_coords[0]
    # Initialize the map class
    obstacle_map = Map(int(robot_radius), int(clearance))
    # Initialize the explorer class
    explorer = Explorer(start_node_coords, goal_node_coords, int(method))
    # Obstacle checking image
    check_image = obstacle_map.check_img.copy()
    # Check validity of start and goal nodes
    if not (check_node_validity(check_image, start_node_coords[1], obstacle_map.height - start_node_coords[0])
            and check_node_validity(check_image, goal_node_coords[1], obstacle_map.height - goal_node_coords[0])):
        print('One of the points lie in obstacle space!!\nPlease try again')
        quit()
    # Get start time for exploration
    start_time = time()
    # Start exploration
    explorer.explore(check_image)
    # Show time for exploration
    print('Exploration Time:', time() - start_time)
    if int(animation) == 1:
        # Get start time for animation
        start_time = time()
        # Display animation of map exploration to find goal
        explorer.show_exploration(obstacle_map.obstacle_img)
        # Show time for animation
        print('Animation Time:', time() - start_time)
