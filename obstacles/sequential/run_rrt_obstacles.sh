#!/bin/bash
g++ -std=c++11 main_obstacles.cpp rrt_obstacles.cpp -o rrt_obstacles
./rrt_obstacles
python3 visualize_rrt_obstacles.py 