from learning_policies.learning_policies import DistributedQLearningAgent, LPIAgent, FQIAgent
import numpy as np
import json

DIST_Q_CONFIG_PATH = "config/DistQ_config.json"
LPI_CONFIG_PATH = "config/LPI_config.json"
FQI_CONFIG_PATH = "config/FQI_config.json"
SEMI_MDP_CONFIG_PATH = "config/semiMDP_reward_config.json"

# given number of agents and algorithm return a list of istances of agents of that specific algorithm
def initialize_agents(n_agents, algorithm, run_num, n_episodes, reward_type, available_actions, agents_connections):
    model_units_folder = f'{n_agents}_units'
    if reward_type == 'reward5':
        with open(SEMI_MDP_CONFIG_PATH) as config_file:
            semiMDP_reward_config = json.load(config_file)
        reward_type += f"/{semiMDP_reward_config['positive_shaping']}_{semiMDP_reward_config['positive_shaping_equal']}_{semiMDP_reward_config['positive_shaping_constant']}_{semiMDP_reward_config['negative_shaping']}_{semiMDP_reward_config['negative_shaping_constant']}_{semiMDP_reward_config['semiMDP_till_end_of_episode']}_IAP500"
    if algorithm == "DistQ":
        with open(DIST_Q_CONFIG_PATH) as config_file:
            config = json.load(config_file)
        return [DistributedQLearningAgent(config, model_units_folder, run_num, i, n_episodes, reward_type, available_actions, agents_connections) for i in range(n_agents)], config['update_values'], config['policy_improvement']
    elif algorithm == "LPI":
        with open(LPI_CONFIG_PATH) as config_file:
            config = json.load(config_file)
        return [LPIAgent(config, model_units_folder, run_num, i, n_episodes, reward_type, available_actions, agents_connections) for i in range(n_agents)], config['update_values'], config['policy_improvement']
    elif algorithm == "FQI":
        with open(FQI_CONFIG_PATH) as config_file:
            config = json.load(config_file)
        return [FQIAgent(config, model_units_folder, run_num, i, n_episodes, reward_type, available_actions, agents_connections) for i in range(n_agents)], config['test_episodes_for_fqi_iteration'], config['multiple_exploration_probabilities'], config['exploration_probabilities'], config['episodes_for_each_explor_for_iteration']
    
# TO-DO: check if move these following 2 functions in DistributedQLearningAgent class

# given a state return an observation which consists of the product that the current agent has
# and of the current skill progress of that specific product (FOR ONLINE TRAINING ONLY)
def get_agent_state_and_product_skill_observation_DISTQ_online(agent, state):
    curr_agent_state = state['agents_state'][agent]
    curr_product_skills = state['products_state'][np.argmax(curr_agent_state)]
    return f'{curr_agent_state}, {curr_product_skills}'

# given a state return an observation which consists of the product that the current agent has
# and of the current skill progress of that specific product (FOR OFFLINE TRAINING ONLY)
def get_agent_state_and_product_skill_observation_DISTQ_offline(state):
    return f'{state["curr_agent_state"]}, {state["curr_product_skills"]}'

