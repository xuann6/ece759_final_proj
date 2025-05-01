#!/bin/bash
g++ -fopenmp -std=c++11 main_obstacles_omp.cpp rrtOmpWithObstacles.cpp rrtStarWithObstacles.cpp rrtBidirectionalWithObstacles.cpp -o rrt_obstacles_omp
