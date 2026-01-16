from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.learning_policies_utils import initialize_agents, get_agent_state_and_product_skill_observation_DISTQ_online
from utils.fqi_utils import get_FQI_state, get_FQI_state_reduced_neighbours_info
from utils.graphs_utils import DistQAndLPIPlotter, FQIPlotter
import json
import numpy as np
import copy
import os

CONFIG_PATH = "config/simulator_config.json"
SEMI_MDP_CONFIG_PATH = "config/semiMDP_reward_config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

test_model = config['test_model']
test_model_name = config['test_model_name']
test_model_n_episodes = config['test_model_n_episodes']
n_agents = config['n_agents']
n_products = config['n_products']
n_runs = config['n_runs']
n_episodes = config['n_episodes']
n_training_episodes = n_episodes
num_max_steps = config['num_max_steps']
n_production_skills = config['n_production_skills']
algorithm = config['algorithm']
custom_reward = config['custom_reward']
available_actions = config['available_actions']
baseline_path = config['baseline_path']
greedy_step = False

if custom_reward == 'reward5':
    with open(SEMI_MDP_CONFIG_PATH) as config_file:
        semiMDP_reward_config = json.load(config_file)

# Dictionary of the paths of all the runs
model_path_runs = {}
baseline_path_runs = {}

for run in range(n_runs):
    if test_model:
        n_episodes = config['n_episodes']

    if algorithm != 'random' and algorithm != 'FQI':
        learning_agents, update_values, policy_improvement = initialize_agents(n_agents, algorithm, run, n_episodes, custom_reward, available_actions, config['agents_connections'])
    elif algorithm == 'FQI':
        learning_agents, test_episodes_for_fqi_iteration, multiple_exploration_probabilities, exploration_probabilities, episodes_for_each_explor_for_iteration = initialize_agents(n_agents, algorithm, run, n_episodes, custom_reward, available_actions, config['agents_connections'])

    if algorithm == 'LPI':
        observations_history_LPI = {}
    
    performance = {}
    greedy_fqi_performance = {}
   
    if algorithm != 'random':
        if test_model:
            # if we are testing load existing model
            for i in range(len(learning_agents)):
                learning_agents[i].load(f'{test_model_name}/run{run}/{i}')
        else:
            # otherwise create model path
            for agent in learning_agents:
                agent.save()
        model_path = learning_agents[0].get_model_name()
    else:
        model_path = baseline_path
        model_path += f'/run{run}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    model_path_runs[run] = model_path
    baseline_path_runs[run] = baseline_path
    baseline_path_runs[run] += f'/run{run}'
    
    env = ProductionPlantEnvironment(config, run, model_path, semiMDP_reward_config)
    
    if test_model:
        n_episodes = test_model_n_episodes

    # if algorithm == 'FQI' and multiple_exploration_probabilities:
    #     actual_expl_prob_index = 0
    #     expl_prob_counter = 0
    #     new_exploration_probability = exploration_probabilities[actual_expl_prob_index]
    #     for fqi_agent in learning_agents:
    #         fqi_agent.change_exploration_probability(new_exploration_probability)
    #     print(f"Exploration probability changed to {new_exploration_probability}")

    for episode in range(n_episodes):
        if algorithm == 'DistQ' or algorithm == 'LPI':
            for agent in learning_agents:
                if test_model:
                    agent.update_exploration_prob(test_model)
                else:
                    # update exploration probability basing on episode
                    agent.update_exploration_prob(test_model, episode + 1)

        if greedy_step:
            greedy_step = False
            for fqi_agent in learning_agents:
                fqi_agent.restore_exploration_probability()
            if multiple_exploration_probabilities:
                actual_expl_prob_index = 0
                expl_prob_counter = 0
                new_exploration_probability = exploration_probabilities[actual_expl_prob_index]
                for fqi_agent in learning_agents:
                    fqi_agent.change_exploration_probability(new_exploration_probability)
                print(f"Exploration probability changed to {new_exploration_probability}")
            else:
                print("Exploration probability restored")
        
        if algorithm == 'FQI' and not test_model and episode % test_episodes_for_fqi_iteration == (test_episodes_for_fqi_iteration - 1):
            greedy_step = True
            for fqi_agent in learning_agents:
                fqi_agent.change_exploration_probability(0)
            print("Exploration probability changed to 0")

        if algorithm == 'FQI' and multiple_exploration_probabilities and not test_model and not greedy_step:
            if expl_prob_counter == episodes_for_each_explor_for_iteration:
                actual_expl_prob_index += 1
                expl_prob_counter = 0
                new_exploration_probability = exploration_probabilities[actual_expl_prob_index]
                for fqi_agent in learning_agents:
                    fqi_agent.change_exploration_probability(new_exploration_probability)
                print(f"Exploration probability changed to {new_exploration_probability}")
            expl_prob_counter += 1

        if algorithm == 'FQI' and not test_model and episode % test_episodes_for_fqi_iteration == 0:
            for fqi_agent in learning_agents:
                fqi_agent.iter()
                fqi_agent.save()

        state = env.reset()
        old_state = copy.deepcopy(state)
        
        for step in range(num_max_steps):
            if algorithm == 'random':
                # TO-DO: remove that after test or put it somewhere else
                if n_products == 1 and config['random_eps_opt']:
                    epsilon = config['random_eps_opt_epsilon']
                    next_skill = [i for i in range(len(state['products_state'][0])) if 1 in state['products_state'][0][i]][0]
                    if np.random.rand() < epsilon:
                        actions = np.array(env.actions)
                        actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
                        action = np.random.choice(actions)
                    else:
                        if n_agents == 9:
                            if state['current_agent'] == 0 and next_skill == 1:
                                action = 9
                            elif state['current_agent'] == 1 and next_skill == 1:
                                action = 9
                            elif state['current_agent'] == 2 and next_skill == 2:
                                action = 10
                            elif state['current_agent'] == 5 and next_skill == 2:
                                action = 10
                            elif state['current_agent'] == 8 and next_skill == 3:
                                action = 8
                            elif state['current_agent'] == 5 and next_skill == 3:
                                action = 8
                            elif state['current_agent'] == 2 and next_skill == 3:
                                action = 11
                            elif state['current_agent'] == 1 and next_skill == 7:
                                action = 11
                            elif state['current_agent'] == 0 and next_skill == 7:
                                action = 10
                            elif state['current_agent'] == 5 and next_skill == 3:
                                action = 8
                            else:
                                actions = np.array(env.actions)
                                actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
                                action = np.random.choice(actions)
                        elif n_agents == 20:
                            if state['current_agent'] == 5 and next_skill == 1:
                                action = 11
                            elif state['current_agent'] == 6 and next_skill == 1:
                                action = 10
                            elif state['current_agent'] == 1 and next_skill == 2:
                                action = 12
                            elif state['current_agent'] == 6 and next_skill == 2:
                                action = 11
                            elif state['current_agent'] == 7 and next_skill == 3:
                                action = 12
                            elif state['current_agent'] == 12 and next_skill == 3:
                                action = 11
                            elif state['current_agent'] == 13 and next_skill == 3:
                                action = 10
                            elif state['current_agent'] == 8 and next_skill == 3:
                                action = 10
                            elif state['current_agent'] == 3 and next_skill == 4:
                                action = 11
                            elif state['current_agent'] == 4 and next_skill == 9:
                                action = 13
                            elif state['current_agent'] == 3 and next_skill == 9:
                                action = 13
                            elif state['current_agent'] == 2 and next_skill == 9:
                                action = 13
                            elif state['current_agent'] == 1 and next_skill == 9:
                                action = 13
                            else:
                                actions = np.array(env.actions)
                                actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
                                action = np.random.choice(actions)
                else:
                    actions = np.array(env.actions)
                    actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
                    action = np.random.choice(actions)
            elif algorithm == 'DistQ':
                agent = learning_agents[state['current_agent']]
                obs = get_agent_state_and_product_skill_observation_DISTQ_online(state['current_agent'], state)
                mask = state['action_mask'][state['current_agent']][n_production_skills:-1]
                action = agent.select_action(obs, mask)
            elif algorithm == 'LPI':
                agent = learning_agents[state['current_agent']]
                obs = agent.generate_observation(state)
                mask = state['action_mask'][state['current_agent']][n_production_skills:-1]
                action = agent.select_action(obs, mask)
            elif algorithm == 'FQI':
                agent = learning_agents[state['current_agent']]
                obs = get_FQI_state({'agents_state': state['agents_state'].copy(), 'products_state': state['products_state'].copy()}, agent.get_observable_neighbours(), n_products)
                # used for lighter neighbours info
                #obs = get_FQI_state_reduced_neighbours_info(state['current_agent'], {'agents_state': state['agents_state'].copy(), 'products_state': state['products_state'].copy()}, agent.get_observable_neighbours(), n_products)
                mask = state['action_mask'][state['current_agent']][n_production_skills:-1]
                action = agent.select_action(obs, mask)

            state, reward, done, truncation, info = env.step(action)
            old_state = copy.deepcopy(state)

            if done:
                print(f"The episode {episode+1}/{n_episodes} is finished.")
                if custom_reward == 'reward5':
                    trajectory_for_semi_MDP = env.get_actual_trajectory()
                    agents_rewards_for_plot = env.get_agents_rewards_for_semiMDP()
                break
            elif truncation:
                print(f"The episode {episode+1}/{n_episodes} has been truncated.")
                if custom_reward == 'reward5':
                    trajectory_for_semi_MDP = env.get_actual_trajectory()
                    agents_rewards_for_plot = env.get_agents_rewards_for_semiMDP()
                break
        
        # semi MDP reward postponed computation (at the end of the episode)
        if custom_reward == 'reward5':
            # semi MDP reward postponed update values (at the end of the episode)
            if not test_model and custom_reward == 'reward5' and algorithm == 'DistQ':
                for actual_step in trajectory_for_semi_MDP:
                    if actual_step['action_selected_by_algorithm']:
                        agent = learning_agents[actual_step['old_state']['current_agent']]
                        action = actual_step['action']
                        reward = actual_step['reward']
                        agent_num = actual_step['old_state']['current_agent']
                        next_agent_num = agent.get_next_agent_number(action)
                        next_agent = learning_agents[next_agent_num]

                        agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation_DISTQ_online(next_agent_num, actual_step['new_state']))
                        obs = get_agent_state_and_product_skill_observation_DISTQ_online(agent_num, actual_step['old_state'])
                        agent.update_values(obs, action, reward, agents_informations)

            elif not test_model and custom_reward == 'reward5' and algorithm == 'LPI':
                for actual_step in trajectory_for_semi_MDP:
                    if actual_step['action_selected_by_algorithm']:
                        agent_num = actual_step['old_state']['current_agent']
                        agent = learning_agents[agent_num]
                        action = actual_step['action']
                        reward = actual_step['reward']
                        if agent_num not in observations_history_LPI.keys():
                            observations_history_LPI[agent_num] = {}
                            observations_history_LPI[agent_num]['O'] = []
                            observations_history_LPI[agent_num]['A'] = []
                            observations_history_LPI[agent_num]['R'] = []

                        observations_history_LPI[agent_num]['O'].append(agent.generate_observation(actual_step['old_state']))
                        observations_history_LPI[agent_num]['A'].append(action)
                        observations_history_LPI[agent_num]['R'].append(reward)

                        if len(observations_history_LPI[agent_num]['O']) > 1:
                            agent.update_values(observations_history_LPI[agent_num]['O'][-2], \
                                                observations_history_LPI[agent_num]['A'][-2], \
                                                observations_history_LPI[agent_num]['R'][-2], \
                                                observations_history_LPI[agent_num]['O'][-1], \
                                                observations_history_LPI[agent_num]['A'][-1])
                            
            if algorithm != 'random' and algorithm != 'FQI':
                if not test_model and update_values == 'episode':
                    for agent in learning_agents:
                        agent.apply_values_update()
                
                if not test_model and policy_improvement == 'episode':
                    for agent in learning_agents:
                        agent.soft_policy_improvement()
        
        if greedy_step and not test_model:
            num = len(greedy_fqi_performance)
            greedy_fqi_performance[num] = {}
            greedy_fqi_performance[num]['episode_duration'] = state['time']
            greedy_fqi_performance[num]['agents_reward_for_plot'] = agents_rewards_for_plot

            for i in range(n_agents):
                mean_reward = np.mean(agents_rewards_for_plot[i])

                greedy_fqi_performance[num][i] = {}
                greedy_fqi_performance[num][i]['mean_reward'] = mean_reward
        else:
            performance[episode] = {}
            performance[episode]['episode_duration'] = state['time']
            performance[episode]['agents_reward_for_plot'] = agents_rewards_for_plot

            for i in range(n_agents):
                mean_reward = np.mean(agents_rewards_for_plot[i])

                performance[episode][i] = {}
                performance[episode][i]['mean_reward'] = mean_reward

    if test_model:
        info_for_plot_path = f"{model_path}/reward_for_plot_test.json"
    else:
        info_for_plot_path = f"{model_path}/reward_for_plot_training.json"
        if algorithm == 'FQI':
            info_for_greedy_plot_path = f"{model_path}/reward_for_greedy_plot_training.json"

    with open(info_for_plot_path, 'w') as outfile:
        json.dump(performance, outfile, indent=6)

    if algorithm == 'FQI' and not test_model:
        with open(info_for_greedy_plot_path, 'w') as outfile:
            json.dump(greedy_fqi_performance, outfile, indent=6)

    if not test_model and algorithm != 'random' and algorithm != 'FQI':
        for agent in learning_agents:
            agent.save()

if test_model:
    if algorithm != 'random' and algorithm != 'FQI':
        plotter = DistQAndLPIPlotter(model_path_runs, baseline_path_runs, n_training_episodes)
        plotter.plot_reward_graphs()
        plotter.plot_performance_graph()

if algorithm == 'FQI' and not test_model:
    plotter = FQIPlotter(model_path_runs, test_episodes_for_fqi_iteration, multiple_exploration_probabilities, exploration_probabilities, episodes_for_each_explor_for_iteration)
    if multiple_exploration_probabilities:
        plotter.plot_performance_graph_multiple_epsilon()
    else:
        plotter.plot_reward_graphs()
        plotter.plot_performance_graph()
        plotter.plot_single_run_performance_graph()
        