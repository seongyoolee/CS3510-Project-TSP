# CS3510-Project-TSP
Timothy Xu (timxu@gatech.edu)
Seon Lee (slee3056@gatech.edu)

*Submitted April 15, 2020*

## Submitted Files
- tsp-3510.py

    Reads input .txt files, algorithm w/ associated methods and logic, processing time, and creates output .txt files
- output-tour.txt 

    Plain text file created by tsp-3510.py containing the cost of the computed TSP tour in the first line and the sequence of node-IDs in the second line.
- algorithm.pdf

    Document describing our algorithm design and the rationale/principles behind our approach 
- README.txt 

    general outline of student info and project components

## Runtime Instructions  
*This project was built with Python-3.*

In the command line, run the program in the following format:

**$: python3 tsp-3510.py <input-coordinates.txt> <output-tour.txt> <time>**

<input-coordinates.txt>: the name of the text input file containing the set of nodes and pairwise distances

<output-tour.txt>: the name of the text file generated containing the cost of the computed TSP tour in the first line and the sequence of node-IDs in the second line.

<time>: maximum number of seconds the program should run

To see debug print statements, pip install termcolor and uncomment the import statement to see print statements in color.

To run the process once, rather than 10 times, change range(10) in main for loop to range(1).
