# Udacity Self-Driving Car Engineer Nanodegree (2019)

This is an archive of the projects I did for obtaining my self-driving car engineer nanodegree in 2019. Here is my [certificate](https://graduation.udacity.com/confirm/AU7N4V).

It was a really fun online program as I learned some key aspects of a self-driving car: Perception, Localization, Planning, and Control. 

On perception, I learned basic computer vision techniques (edge detection, space transforms, thresholding, distortion removal, perspective transform) as well as using deep learning models to classify image data (e.g. traffic sign recognition). Furthermore, I also learned how a car could perceive its surroundings from sensor fusion (tracking other objects it sees e.g. through LIDAR and RADAR data)

Then, on localization, I learned localization, leveraging the Markov model (Markov localization) and how this can be done with Bayesian filters (e.g. particle filters). It was really interesting to see how Bayes filters come in many form (e.g. histogram filters, Kalman filters) yet they share the same fundamental principles (computing posteriors for a given prior based on some transition model).

On planning, I learned how data-driven and model-based approaches could be used to anticipate/predict the behavior of surrounding subjects (we focus on cars) and generate trajectories for our self-driving car accordingly.

I also learned how to control the outputs of the car (e.g. torque) using PID control. It is certainly a basic technique compared to many more advanced control techniques that have been developed (e.g. MPC, RL), yet it is still extremely reliable and widely used in many systems even in this day and age.

Finally, I worked on a capstone project in ROS that essentially covers most of the things I learned by writing these subsytems as ROS nodes and allowing them to communicate with one another via ROS topics.

Note: In each project folder you will see a README file describing the project. When i was doing it in 2019 (I was quite new to coding, ML, and robotics back then), I wrote READMEs only for some projects. Thus, recently I went to review the other projects and wrote READMEs as best as I can describing what I remember doing, especially highlighting the theoretical aspects based on what I learned in the lectures. 




