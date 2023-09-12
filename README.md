## Development of Python Library for the Geodesic Dome
This project aims to complete a python library implementing 3 key features
related to the Geodesic Dome:

- Create a Geodesic Dome with the ability to continuously tessellate each face
  according to a specified frequency
- Tessellate a subset of faces
- Implement a neighbour search algorithm to find all points within a specified
  distance, where each neighbour can be found in O(1) time

We aim to implement these three features correctly while also trying to make the
code performant both in terms of time and memory.

## Overview
I completed this project with 5 other team members as a part of a university capstone project. We had a chance to explore the Python library Numba used to optimize Python code. The library and docs can be found at this link: https://pypi.org/project/geodome/. The most challenging part of this assignment was understanding how to present arrays in ways that were compliant with Numba. During the development process we ran into many issues with array manipulation. 
