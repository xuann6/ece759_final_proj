#!/bin/bash
g++ -fopenmp -std=c++11 main_obstacles_omp.cpp rrtObstaclesOmp.cpp -o rrt_obstacles_omp
