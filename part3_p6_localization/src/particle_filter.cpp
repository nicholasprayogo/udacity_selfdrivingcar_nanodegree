#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;

//   // TODO: Set standard deviations for x, y, and theta
//   std_x = std[0];
//   std_y = std[1];
//   std_theta = 0.05;

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);

  // TODO: Create normal distributions for y and theta
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {

    // TODO: Sample from these normal distributions like this:
    //   sample_x = dist_x(gen);
    //   where "gen" is the random engine initialized earlier.
    Particle new_p; // use the struct in particle_filter.h
    new_p.id = int(i);
    new_p.weight = 1.0;
    new_p.x = dist_x(gen);
    new_p.y= dist_y(gen);
    new_p.theta= dist_theta(gen);

    particles.push_back(new_p);
  }
  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (size_t i = 0; i < num_particles; ++i) {
        // collect old values
        double x_old	 = particles[i].x;
        double y_old	 = particles[i].y;
        double theta_old = particles[i].theta;
//     	initialize new values
        double theta_pred, x_pred, y_pred;
    	if (abs(yaw_rate)>1e-5) {
            // turning motion model
            theta_pred = theta_old + yaw_rate * delta_t;
            x_pred	   = x_old + velocity / yaw_rate * (sin(theta_pred) - sin(theta_old));
            y_pred	   = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_pred));
        } else {
            // going straight motion model
            theta_pred = theta_old;
            x_pred	   = x_old + velocity * delta_t * cos(theta_old);
            y_pred	   = y_old + velocity * delta_t * sin(theta_old);
        }

        // use pred values as mean for normal distrib
        normal_distribution<double> dist_x(x_pred, std_pos[0]);
        normal_distribution<double> dist_y(y_pred, std_pos[1]);
        normal_distribution<double> dist_theta(theta_pred, std_pos[2]);

        // update particle with noisy prediction
        particles[i].x	   = dist_x(gen);
        particles[i].y	   = dist_y(gen);
        particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights   **/
  for (auto& obs : observations) {
        double min_dist = numeric_limits<double>::max();
        for (const auto& pred_obs : predicted) {
            double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
            if (d < min_dist) {
                obs.id	 = pred_obs.id;
                min_dist = d;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
	double var_x = pow(std_landmark[0], 2);
	double var_y = pow(std_landmark[1], 2);
	double covar_xy = std_landmark[0] * std_landmark[1];

  	for (int i=0; i < num_particles; i++) {

      Particle& particle = particles[i];
      long double weight = 1.0;
      
	  // transform to map coords
      vector<LandmarkObs> observed_landmarks_map_ref;
      for (LandmarkObs obs : observations) {
        LandmarkObs transformed_obs;
        transformed_obs.x =  particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta) ;
		transformed_obs.y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta) ;
        observed_landmarks_map_ref.push_back(transformed_obs);
      }
      
      // compare each obs (measurement) with each pred
      vector<LandmarkObs> pred_landmarks;
      for (const auto& landmark : map_landmarks.landmark_list ) {
          double distance;
//           distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
          distance = fabs(particle.x- landmark.x_f) + fabs(particle.y - landmark.y_f);
          if (distance < sensor_range) {
            		LandmarkObs pred_lmark;
            		pred_lmark.id = landmark.id_i;
           			pred_lmark.x = landmark.x_f;
            		pred_lmark.y = landmark.y_f;
           			pred_landmarks.push_back(pred_lmark);
				}
        }
      
      dataAssociation(pred_landmarks, observed_landmarks_map_ref);
      
      // update weights using the product formula
      for (const auto& transformed_obs : observed_landmarks_map_ref){
        LandmarkObs nearest;

        for (const auto& nearestland: pred_landmarks)
          	if (transformed_obs.id == nearestland.id){
              nearest.x = nearestland.x;
              nearest.y = nearestland.y;
              break;
        	}
        double delta_x = transformed_obs.x- nearest.x;
        double delta_y = transformed_obs.y- nearest.y;
        double num = exp(-0.5*( pow(delta_x,2.0)/var_x + pow(delta_y,2.0)/var_y));
        double denom = 2*M_PI*covar_xy;
        weight *= num/denom;
      }

      // update particle weight
      particles[i].weight = weight;

    }
  
    double norm_factor = 0.0;
    for (const auto& particle : particles)
        norm_factor += particle.weight;
    // normalize
    for (auto& particle : particles)
        particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distr    double norm_factor = 0.0;
    for (const auto& particle : particles)
        norm_factor += particle.weight;

    // Normalize weights s.t. they sum to one
    for (auto& particle : particles)
        particle.weight /= (norm_factor + numeric_limits<double>::epsilon());ibution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
    vector<double> particle_weights;
    for (const auto& particle : particles)
        particle_weights.push_back(particle.weight);

    discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

    vector<Particle> resampled_particles;
    for (size_t i = 0; i < num_particles; ++i) {
        int k = weighted_distribution(gen);
        resampled_particles.push_back(particles[k]);
    }

    particles = resampled_particles;

    // reset
    for (auto& particle : particles)
        particle.weight = 1.0;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
