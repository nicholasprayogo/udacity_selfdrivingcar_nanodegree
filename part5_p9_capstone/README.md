# Capstone Project: Programming a Real Self-Driving Car

This capstone project combines everything learned in the previous projects (computer vision, sensor fusion, localization, path planning, and control). 

## Background

Please refer to the other projects' READMEs for some theoretical and practical description of the subsystems implemented. 

## Architecture

<img src="./imgs/architecture.png " alt="drawing" width="1000"/>

The main theme of this project is designing the architecture such that each subsystem could listen to one another. This is made possible by writing and configuring the ROS nodes and topics for publishing and subscribing. 

Further information on code execution can be found in `./ros/src/README.md` (provided by Udacity)
