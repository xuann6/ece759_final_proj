#!/bin/bash

# Compile and run RRT with OpenMP
g++ -std=c++11 -fopenmp ../simple/rrt.cpp ../simple/rrtStar.cpp rrtOmp.cpp rrtStarOmp.cpp rrtBiOmp.cpp rrtInformedOmp.cpp main.cpp -o rrt_omp_program && ./rrt_omp_program
