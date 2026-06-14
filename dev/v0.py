"""
This example shows how to define agents so that they can be systematically run
in a specific environment (important for agent evaluation purposes)

"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude

# class DevAgent(BaseAgent):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args)

#     self.launch_time = kwargs.get("launch_time", 0.0)

#   def get_action(self, observation):
    
#      return {
#         "launch": (observation["simulation_time"] >= self.launch_time),
#         "launch_inclination_heading": np.array([90, 0]),
#         "tvc": np.array([0.0001, 0.0]),
#         "roll": 0,
#         "throttle": 1,
#     }



class DevAgent(BaseAgent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args)

    self.launch_time = kwargs.get("launch_time", 0.0)

    self.rate_targets = np.array(kwargs.get("rate_targets", [0.0, 0.0, 0.0]))
    self.rate_KP      = np.array(kwargs.get("rate_KP", [100.0, 100.0, 100.0]))
    self.rate_KI      = np.array(kwargs.get("rate_KI", [0.0, 0.0, 0.0]))
    self.rate_KD      = np.array(kwargs.get("rate_KD", [0.0, 0.0, 0.0]))
    self.rate_errors  = np.zeros((3,1))

  def get_action(self, observation):
    
    if not np.isnan(observation["rocket_sensors"][:3]).any():

      sensor_frequency = self.given_parameters["rocket"]["sensors"]["sampling_rate"]

      self.rate_errors = np.append(
        self.rate_errors,
        (self.rate_targets - observation["rocket_sensors"][:3]).reshape(-1, 1),
        axis=1
      )

      rate_errors_integral = (
        np.sum(self.rate_errors, axis=1) / sensor_frequency
      )
      
      rate_errors_derivative = (
        (self.rate_errors[:, -1] - self.rate_errors[:, -2]) * sensor_frequency
        if self.rate_errors.shape[1] > 1
        else np.array([0.0, 0.0, 0.0])
      )

      tvc = (
        self.rate_KP * self.rate_errors[:, -1] +
        self.rate_KI * rate_errors_integral[:3] +
        self.rate_KD * rate_errors_derivative[:3]
      )

    else:
      tvc = np.array([0.0, 0.0, 0.0])
    
    return {
        "launch": (observation["simulation_time"] >= self.launch_time),
        "launch_inclination_heading": np.array([90, 0]),
        "tvc": np.array([tvc[0], tvc[1]]),
        "roll": tvc[2],
        "throttle": 1,
    }