# Sub-Optimal-Path-Finder
[![Build Status](https://travis-ci.org/urastogi885/sub-optimal-path-finder.svg?branch=master)](https://travis-ci.org/github/urastogi885/sub-optimal-path-finder)
[![License](https://img.shields.io/badge/License-MIT--Clause-blue.svg)](https://github.com/urastogi885/sub-optimal-path-finding/blob/master/LICENSE)

## Overview
This project implements search-based algorithms that yield a sub-optimal path if one exists. Path finding 
algorithms, such as Weighted A*, have been used find a sub-optimal path for the robot from the user-defined start to end 
point. The project also generates animation to visualize the exploration of each of the mentioned search-based methods. 
It first checks that the user inputs do not lie in the obstacle space. Note that the obstacle space is pre-defined 
and static.

In Weighted A*, the heuristic cost is multiplied by a weight factor, epsilon. If epsilon is 1, Weighted A* becomes A*.

Unlike A* and Weighted A*, Depth-first search(DFS) has to concept of cost-to-come and cost-to-goal. Moreover, it is 
based on a Last-In-First-Out (LIFO) queue. A LIFO queue is simply known as a stack.

<p align="center">
  <img src="https://github.com/urastogi885/sub-optimal-path-finder/blob/master/images/faster.gif">
  <br><b>Figure 1 - Comparison of A*, Weighted A*, and DFS</b><br>
</p>

## Todo

- Implement D*
- Implement ARA*
- Implement AD*

## Dependencies

- Python3
- Python3 Libraries: Numpy, OpenCV-Python, Math, Queue, Time, Sys, Ast

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)
````
sudo apt install python3
sudo apt install python3-pip
pip3 install numpy opencv-python queue
````

- Check if your system successfully installed all the dependencies
- Open terminal using Ctrl+Alt+T and enter python3
- The terminal should now present a new area represented by >>> to enter python commands
- Now use the following commands to check libraries: (Exit python window using Ctrl+Z if an error pops up while
running the below commands)
````
import numpy
import cv2
import queue
````

## Run

- Using the terminal, clone this repository and go into the project directory, and run the main program:
````
git clone https://github.com/urastogi885/sub-optimal-path-finder
cd sub-optimal-path-finder
````
- If you have a compressed version of the project, extract it, go into project directory, open the terminal, and run
the robot explorer.
````
python3 robot_explorer.py <start_x,start_y> <goal_x,goal_y> <robot_radius> <clearance> <method> <animation>
````
- An example for using *Weighted A** for a rigid robot:
````
python3 robot_explorer.py 50,30 150,150 2 1 0 1
````
- The output in the [Overview](#overview) section has been generated using the above arguments.
