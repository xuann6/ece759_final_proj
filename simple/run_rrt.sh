#!/bin/bash

# Compile and run RRT
g++ -std=c++11 rrt.cpp rrtStar.cpp rrtInformed.cpp rrtBidirectional.cpp main.cpp TimerUtils.cpp -o rrt_program && ./rrt_program