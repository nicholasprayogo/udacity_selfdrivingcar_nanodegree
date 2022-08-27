## Model Documentation: Path Planning Project 
### Compliance with Rubric Specifications
1. The car is able to drive at least 4.32 miles without incident. 

When the simulator is run with my code, the car was able to drive at over 5 miles without incident.
2. The car drives according to the speed limit.

This was possible due to the fact that I only accelerate if the reference velocity I set remains below 49.5 mph.

3. Max Acceleration and jerk are not exceeded.

Since I initialize the car's reference velocity as 0.0 mph and only increasing it through gradual accelerations of 0.4 mph increments, the car never exceeds the max acceleration of 10 m/s^2.

4. Car does not have collisions.

By checking if there is another car in the same lane of my car within 30 m longitudinally, the car decreases its speed by 0.4 mph gradually whenever that condition is met. Hence, it is really unlikely that it collides into a car in this simulation, but problem will be present if in real life a sports car, for instance, is driving at 200 km/h and brakes completely all of a sudden. This could be taken into account for future iterations of this project.

5. Car stays in lane, except for the time between changing lanes.

By ensuring d remains at 4*(lane+0.5), the car is able to stay in lane.

6. The car is able to change lanes.
This is pretty straightforward given the ease of use of the sensor fusion and car state data provided in the code, where by changing d, car is able to change lanes. The smoothness of the lane changes, however, is more challenging and achieved by incorporating the spline library for creating a piecewise function fit over the car's trajectory. In future iterations, the project will include cost functions for generating and choosing trajectory splines with least jerk, least distance from center, etc. 
