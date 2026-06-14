import matplotlib.pyplot as plt
import numpy as np

import sys
import importlib

if len(sys.argv) < 2:
  raise ValueError(
    "Configuration file path is required. "
    "Usage: python run.py <agent_name>"
  )
path = "dev."+sys.argv[1]
DevAgent = importlib.import_module(path).DevAgent

scenario_number = "empty"
agent_name = "Dev Agent"

# These are rads not degs
agent_kwargs = {
  "launch_time": 0.0,
  "rate_targets": [0.1, 0.0, 0.0],
  "rate_KP": [1000.0, 100.0, 100.0],
  "rate_KI": [0.0, 0.0, 0.0],
  "rate_KD": [0.0, 0.0, 0.0]
}

def run_for_development():
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

    # Load scenario parameters
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

    # Create environment with scenario parameters turn off rendering to make own plots
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = DevAgent(given_parameters, **agent_kwargs)

    # use seed=None to randomize environment
    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False

    angular_rates = np.full((3, 1), np.nan)
    time = np.full(1, np.nan)
    height = np.full(1, np.nan)

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)

        # ground truth angular rates, should not pass to agent
        angular_rates = np.append(angular_rates, info["rocket_states"][10:13].reshape(-1, 1), axis=1)
        time = np.append(time, observation["simulation_time"])
        height = np.append(height, info["rocket_states"][2])

        print(f"simulation_time: {observation['simulation_time']:.2f} sec, reward: {reward:.2f}", end='\r')

    plt.subplot(2, 1, 1)
    plt.plot(time, angular_rates[0], 'r-', label='x_rate')
    # plt.plot(time, angular_rates[1], 'g-', label='y_rate')
    # plt.plot(time, angular_rates[2], 'b-', label='z_rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Rates (rad/s)')
    # plt.xlim(0, 30)
    # plt.ylim(-0.1, 0.1)
    plt.legend()

    np.savetxt("dev/data.csv", np.asarray([time, angular_rates[0]]), delimiter=",")

    plt.subplot(2,1,2)
    plt.plot(time, height, 'g-', label='height')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Rates (rad/s)')
    plt.legend()


    # # TVC controller observed variables are tuples: (time, gimbal_x, gimbal_y)
    # tvc = env._rocket_flight.rocket._controllers[0].observed_variables
    # tvc_array = np.array(tvc, dtype=float)
    # plt.subplot(2, 1, 2)
    # plt.plot(tvc_array[:, 0], tvc_array[:, 1], 'r-', label='tvc_x')
    # plt.plot(tvc_array[:, 0], tvc_array[:, 2], 'b-', label='tvc_y')
    # plt.xlabel('Time (s)')
    # plt.ylabel('TVC Gimbal Angle (deg)')
    # # plt.xlim(0, 30)
    # # plt.ylim(-0.1, 0.1)
    # plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Scenario {scenario_number} evaluation completed with agent '{agent_name}'.")
    print(f"Final reward: {reward}")

    # env._rocket_flight.all_info() # Uncomment to print all info from RocketPy

def run_for_evaluation():
    from BalloonPoppingGymEnv.evaluation.evaluate import evaluate_scenario

    # Load agent class dynamically from specified module path.
    # Equivalent to run command: python evaluate.py <path_to_eval_config.yaml>
    evaluate_scenario(
        DevAgent,
        agent_kwargs=agent_kwargs,
        agent_name=agent_name,
        scenario_number=scenario_number,
        render_mode='matplotlib',
    )

if __name__ == "__main__":
    # Use this function for development and debugging purposes.
    run_for_development()

    # Use this function for evaluation purposes.
    # run_for_evaluation()