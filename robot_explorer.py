# Import necessary standard libraries
import ast
from sys import argv
from time import time
# Import necessary custom-built classes and methods
from utils.obstacle_space import Map


"""
Add various parameters as input arguments from user
:param start_node_data: a tuple of 2 values: start coordinates
:param goal_node_data: a tuple of 2 values: goal coordinates
:param robot_radius: radius of the robot
:param clearance: minimum distance between robot and any obstacle 
:param method: d for dijkstra or a for a-star
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
    # Obstacle checking image
    check_image = obstacle_map.check_img.copy()
