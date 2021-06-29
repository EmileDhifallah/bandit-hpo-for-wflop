from bandit_hpo import helper_functions

# wflopg library
import wflopg
import wflopg.optimizers as opt

# visualizing
import matplotlib.pyplot as plt

# basic helper libraries
import time
import itertools
import os
import random
import math

# data manipulation libraries
import numpy as np
import pandas as pd


class ExperimentHPO:
    '''The main hpo experiment object.
       Using this object the different HPO algorithms can be called on
       the problem file of a specific wind farm layout optimization problem.
    '''

    def __init__(self, wflop_problem):        # wflop_problem should consist of entire file path!
        self.wflop_problem = wflop_problem
        os.chdir(os.path.split(wflop_problem)[0])


    def fast_run(self, max_iterations=10, method='s', multiplier=1.0, scaling=[0.8, 1.2]):
        '''Runs step-based optimization on given wflop problem file with manual hyperparameter selection;

            input: problem data file and hyperparameter settings;
            output: final farm wake loss.
            '''

        # initializing owflop object and running step iterator
        o = wflopg.Owflop()
        o.load_problem(self.wflop_problem)
        o.calculate_wakeless_power()
        optim_result = opt.step_iterator(o, max_iterations=max_iterations, methods=method,
                                         multiplier=multiplier, scaling=scaling, visualize=True)

        # storing step objective values and printing these
        objective_values = optim_result['objective'].values
        print()
        print('Wake farm loss development over', len(objective_values), 'iterations:')
        print(objective_values)
        final_wake = objective_values[-1]
        iterations = [num for num in range(len(objective_values))]

        plt.plot(iterations, objective_values)
        plt.xlabel('iteration')
        plt.ylabel('total wake loss')
        plt.show()

        print('Total wake loss:')
        return final_wake

    def grid_search(self, max_iterations=[100], method=['abcs'], multiplier=[0.5, 1.0, 1.5],
                    scaling=[[0.5, 1.5], [0.8, 1.2]], visualize=True):
        '''Performs a grid search on all combinations of given h.p. settings;

            input:  -problem data file;
                    -list of all h.p. values to consider, for all hyperparameters;
            output: Pandas DataFrame containing optimal configuration and corresponding
                    final farm wake loss.
        '''

        result_storage = []

        # creating configuration grid
        hyperparameter_combinations = itertools.product(max_iterations, method, multiplier, scaling)

        data_dict = {}

        wake_progression = []
        axes = []

        start_time = time.time()
        total_time = start_time
        print('Runtime counter started')
        print('Total runtime: ', "-- %s seconds --" % (start_time))

        iteration = 0

        # iterating over all configurations
        for run in hyperparameter_combinations:
            # initializing owflop object
            o = wflopg.Owflop()
            o.load_problem(self.wflop_problem)
            o.calculate_wakeless_power()

            iteration += 1
            axes.append(str(run))

            print('Running algorithm for hyperparameter combination:')
            print('Max iterations: ', run[0])
            print('Method: ', run[1])
            print('Multiplier:', run[2])
            print('Scaling:', run[3])

            start_iter_time = time.time()

            # running optimization process and storing result
            optim_result = opt.step_iterator(o, max_iterations=run[0], methods=run[1],
                                             multiplier=run[2], scaling=run[3], visualize=False)
            final_objective_value = optim_result['objective'].values[-1]
            wake_progression.append(final_objective_value)

            end_iter_time = time.time()
            total_iter_time = end_iter_time - start_iter_time
            total_time = total_time + total_iter_time
            print('Runtime iteration %s : %s' % (iteration, total_iter_time))
            print('Total runtime: %s' % total_time)

            # saving results
            if not bool(data_dict):
                data_dict['params'] = run
                data_dict['objective'] = final_objective_value

            # checking if current configuration is optimal one
            if bool(data_dict):
                if final_objective_value < data_dict['objective']:
                    data_dict['params'] = run
                    data_dict['objective'] = final_objective_value

            result_storage.append([self.wflop_problem, run[2], run[3], run[1], run[0], final_objective_value, 0])

        end_time = time.time()
        total_runtime = end_time - start_time
        print('\n', 'Progression of wake value over iterations:', '\n')
        print(wake_progression)

        # visualize total grid's performance
        if visualize:
            print('\n', 'All runs summarized:', '\n')
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(axes, wake_progression)
            plt.xticks(rotation=90)
            plt.show()

        print('\n', 'Smallest amount of wake reached with hyperparameter combination:', '\n')  # print parameters
        print('Max iterations: ', data_dict['params'][0])
        print('Method: ', data_dict['params'][1])
        print('Multiplier:', data_dict['params'][2])
        print('Scaling:', data_dict['params'][3])

        # creating results dataframe and saving df to csv
        results = pd.DataFrame(result_storage, columns=['File name', 'Multiplier', \
                                                        'Scaling', 'Method', 'Max. Iterations', 'Objective Value Run',
                                                        'Runtime In Seconds'])

        if visualize:
            helper_functions.grid_search_heatmaps(results, self.wflop_problem, multiplier, scaling)

        result_name = self.wflop_problem + '-method-' + method[0]
        results.to_csv(r'C:\Users\20175862\Documents\BEP\Code tests\heatmap_experiments\%s.csv' % result_name)

        print('Final objective value summary: ')
        return results

    def k_times_random_search(self, max_iterations=[100], method=['abcs'], multiplier=['1.0'],
                              scaling=[[0.8, 1.2]], k=10, visualize=True):
        '''Performs random search for k configurations given a search space;

            input:-self.wflop_problem: (string) file path;
                  -max_iterations: range [a,b] where 0<=a<=b;
                  -method: list of at least one and at most all combinations of the strings in {'s','a','b','c','m'};
                  -multiplier: range [a,b], where 0<=a<=b;
                  -scaling: ranges [a,b,c,d]; where 0<=a<=b<=c<=d; a and b create the range for the lower scaling bound
                   and c and d for the upper scaling bound;
                  -k: integer for number of configurations;

            output: Pandas DataFrame containing optimal configuration and corresponding
                    final farm wake loss.
            '''
        # intializing wflop object
        o = wflopg.Owflop()
        o.load_problem(self.wflop_problem)
        o.calculate_wakeless_power()

        data_dict = {}

        wake_progression = []
        axes = []

        start_time = time.time()
        print('Runtime counter started')
        print('Total runtime: ', "-- %s seconds --" % (start_time))

        # looping over k
        for i in range(k):
            # making sure hyperparameter ranges are in required format
            # and creating uniformely sampled configuration
            iteration = i + 1

            if max_iterations[0] <= max_iterations[1] and len(max_iterations) == 2:
                param_max_iterations = math.ceil(random.uniform(max_iterations[0], max_iterations[1]))
            else:
                raise ValueError("Wrong parameter input. There should be 2 float or integer bounds in max_iterations, \
                                    where the upper bound has to be greater than or equal to the lower bound")

            if set(method).issubset(set(['a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'])):
                param_method = method[random.randint(0, len(method) - 1)]
            else:
                raise ValueError("Wrong parameter input. The method list can only obtain strings from the  set \
                                 {'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'}")

            if multiplier[0] <= multiplier[1] and len(multiplier) == 2:
                param_multiplier = random.uniform(multiplier[0], multiplier[1])
            else:
                raise ValueError("Wrong parameter input. There should be 2 float or integer bounds in multiplier, where \
                                 the upper bound has to be greater than or equal to the lower bound")

            if scaling[0] <= scaling[1] and scaling[2] <= scaling[3] and len(scaling) == 4:
                scaling_lower_bound = random.uniform(scaling[0], scaling[1])
                if scaling_lower_bound > scaling[2]:
                    scaling_upper_bound = random.uniform(scaling[2], scaling[3])
                else:
                    scaling_upper_bound = random.uniform(scaling_lower_bound, scaling[3])
                param_scaling = [scaling_lower_bound, scaling_upper_bound]
            else:
                raise ValueError("Wrong parameter input. There should be 4 float or integer numbers in scaling: \
                                 a lower and upper bound for the lower scaling bound and a lower and upper bound \
                                 for the upper scaling bound, where the upper bounds are greater than or equal to \
                                 its respective lower bounds")

            param_list = [param_max_iterations, param_method, param_multiplier, param_scaling]
            axes.append(str(param_list))

            print('\n')
            print('Running algorithm for hyperparameter combination:')
            print('Max iterations: ', param_list[0])
            print('Method: ', param_list[1])
            print('Multiplier:', param_list[2])
            print('Scaling:', param_list[3])

            start_iter_time = time.time()

            # running optimizer for current configuration
            optim_result = opt.step_iterator(o, max_iterations=param_list[0], methods=param_list[1],
                                             multiplier=param_list[2], scaling=param_list[3], visualize=True)
            final_objective_value = optim_result['objective'].values[-1]
            wake_progression.append(final_objective_value)

            end_iter_time = time.time()
            total_iter_time = end_iter_time - start_iter_time
            total_time = total_time + total_iter_time
            print('Runtime iteration %s : %s' % (iteration, total_iter_time))
            print('Total runtime: %s' % total_time)

            # saving optimization results in dictionary
            if not bool(data_dict):
                data_dict['params'] = param_list
                data_dict['objective'] = final_objective_value

            # checking if current configuration is best one yet
            if bool(data_dict):
                if final_objective_value < data_dict['objective']:
                    data_dict['params'] = param_list
                    data_dict['objective'] = final_objective_value

        print('\n Different wake values over iterations:', '\n')
        print(wake_progression)

        # visualizing all objective values
        if visualize:
            print('\n', 'All runs summarized:', '\n')
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(axes, wake_progression)
            plt.xticks(rotation=90)
            plt.show()

        print('Smallest amount of wake reached with hyperparameter combination:', '\n')  # print parameters
        print('Max iterations: ', data_dict['params'][0])
        print('Method: ', data_dict['params'][1])
        print('Multiplier:', data_dict['params'][2])
        print('Scaling:', data_dict['params'][3])

        print('Final objective value: ', data_dict['Objective'])

        # creating results dataframe and saving df to csv
        end_time = time.time()
        total_runtime = end_time - start_time
        result_storage = [[self.wflop_problem, data_dict['params'][2], data_dict['params'][3], \
                           data_dict['params'][1], data_dict['params'][0], data_dict['objective'], total_runtime]]

        results = pd.DataFrame(result_storage, columns=['File name', 'Multiplier', 'Scaling', 'Method', \
                                                        'Max. Iterations', 'Objective Value Run', 'Runtime In Seconds'])
        result_name = self.wflop_problem + '-method-' + method[0]
        results.to_csv(r'C:\Users\20175862\Documents\BEP\Code tests\heatmap_experiments\%s.csv' % result_name)

        return results

    def time_bound_random_search(self, max_iterations=[100], method=['abcs'], multiplier=[1.0],
                                 scaling=[[0.8, 1.2]], time_constraint=100, visualize=True):
        '''Performs random search for x amount of resources (measured in seconds) given a search space;

            input:-self.wflop_problem: (string) file path;
                  -max_iterations: range [a,b] where 0<=a<=b;
                  -method: list of at least one and at most all combinations of the strings in {'s','a','b','c','m'};
                  -multiplier: range [a,b], where 0<=a<=b;
                  -scaling: ranges [a,b,c,d]; where 0<=a<=b<=c<=d; a and b create the range for the lower scaling bound
                   and c and d for the upper scaling bound;
                  -time_constraint: integer for number of seconds;

            output: Pandas DataFrame containing optimal configuration and corresponding
                    final farm wake loss.
            '''

        start_time = time.time()
        # initializing wflop model
        o = wflopg.Owflop()
        o.load_problem(self.wflop_problem)
        o.calculate_wakeless_power()

        data_dict = {}

        wake_progression = []
        axes = []
        iteration = 0

        start_time = time.time()
        total_time = start_time - start_time
        print('Runtime counter started')
        print('Total runtime: ', "-- %s seconds --" % (total_time))

        # keep iterating over new random configurations while total time is less than allowed time
        while total_time < time_constraint:
            iteration = iteration + 1

            # uniformely sample new configuration
            if max_iterations[0] <= max_iterations[1] and len(max_iterations) == 2:
                param_max_iterations = math.ceil(random.uniform(max_iterations[0], max_iterations[1]))
            else:
                raise ValueError("Wrong parameter input. There should be 2 float or integer bounds in max_iterations, \
                                 where the upper bound has to be greater than or equal to the lower bound")

            assert False not in (x in ['a', 'b', 'c', 's', 'm'] for x in method[0]), "Wrong parameter input. \
                                 The method list can only obtain strings from the  set \
                                 {'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'}"

            param_method = method[random.randint(0, len(method) - 1)]

            if multiplier[0] <= multiplier[1] and len(multiplier) == 2:
                param_multiplier = random.uniform(multiplier[0], multiplier[1])
            else:
                raise ValueError("Wrong parameter input. There should be 2 float or integer bounds in multiplier, where \
                                 the upper bound has to be greater than or equal to the lower bound")

            if scaling[0] <= scaling[1] and scaling[2] <= scaling[3] and len(scaling) == 4:
                scaling_lower_bound = random.uniform(scaling[0], scaling[1])
                if scaling_lower_bound > scaling[2]:
                    scaling_upper_bound = random.uniform(scaling[2], scaling[3])
                else:
                    scaling_upper_bound = random.uniform(scaling_lower_bound, scaling[3])
                param_scaling = [scaling_lower_bound, scaling_upper_bound]
            else:
                raise ValueError("Wrong parameter input. There should be 4 float or integer numbers in scaling: \
                                 a lower and upper bound for the lower scaling bound and a lower and upper bound \
                                 for the upper scaling bound, where the upper bounds are greater than or equal to \
                                 its respective lower bounds")

            param_list = [param_max_iterations, param_method, param_multiplier, param_scaling]
            axes.append(str(param_list))

            print('\n')
            print('Running algorithm for hyperparameter combination:')
            print('Max iterations: ', param_list[0])
            print('Method: ', param_list[1])
            print('Multiplier:', param_list[2])
            print('Scaling:', param_list[3])

            start_iter_time = time.time()

            # run optimizer for current configuration and storing objective values
            optim_result = opt.step_iterator(o, max_iterations=param_list[0], methods=param_list[1], \
                                             multiplier=param_list[2], scaling=param_list[3], visualize=True)
            final_objective_value = optim_result['objective'].values[-1]
            wake_progression.append(final_objective_value)

            end_iter_time = time.time()
            total_iter_time = end_iter_time - start_iter_time
            total_time = total_time + total_iter_time
            print('Runtime iteration %s : %s' % (iteration, total_iter_time))
            print('Total runtime: %s' % total_time)

            # storing final objective value
            if not bool(data_dict):
                data_dict['params'] = param_list
                data_dict['objective'] = final_objective_value

            # checking if current configuration is best yet
            if bool(data_dict):
                if final_objective_value < data_dict['objective']:
                    data_dict['params'] = param_list
                    data_dict['objective'] = final_objective_value

        print('%s runs performed in %s seconds' % (iteration, total_time), '\n')

        # checking if total time stayed inside or
        # went over allowed total time (over-limit time is allowed for last iteration)
        if total_time > time_constraint:
            print('The final iteration took %s seconds more than the time limit of %s seconds' % \
                  (total_time - time_constraint, time_constraint), '\n')

        print('Different wake values over runs:', '\n')
        print(wake_progression)

        # visualizing all objective values
        if visualize:
            print('\n')
            print('All runs summarized:', '\n')
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(axes, wake_progression)
            plt.xticks(rotation=90)
            plt.show()

        print('\n')
        print('Smallest amount of wake reached with hyperparameter combination:', '\n')  # print parameters
        print('Max iterations: ', data_dict['params'][0])
        print('Method: ', data_dict['params'][1])
        print('Multiplier:', data_dict['params'][2])
        print('Scaling:', data_dict['params'][3])

        # retrieving total time and storing of results
        end_time = time.time()
        total_runtime = end_time - start_time
        result_storage = [[self.wflop_problem, data_dict['params'][2], data_dict['params'][3], \
                           data_dict['params'][1], data_dict['params'][0], data_dict['objective'], total_runtime]]

        print('Final objective value: ')

        # creating results dataframe and saving df to csv
        results = pd.DataFrame(result_storage,
                               columns=['File name', 'Multiplier', 'Scaling', 'Method', \
                                        'Max. Iterations', 'Objective Value Run', 'Runtime In Seconds'])

        result_name = self.wflop_problem + '-method-' + method[0]
        results.to_csv(r'C:\Users\20175862\Documents\BEP\Code tests\heatmap_experiments\%s.csv' % result_name)

        return results

    def successive_halving(self, method=['abcs'], multiplier=[0.1], scaling=[[0.8,1.2]],
                           num_configurations=10, budget=100):
        '''Performs successive halving preset number of configurations and budget given a search space;

            input:-self.wflop_problem: (string) file path
                  -method: list of at least one and at most all combinations of the strings in {'s','a','b','c','m'};
                  -multiplier: range [a,b], where 0<=a<=b;
                  -scaling: ranges [a,b,c,d]; where 0<=a<=b<=c<=d; a and b create the range for the lower scaling bound
                   and c and d for the upper scaling bound;
                  -num_configurations: integer number of random configurations to start with
                  -budget: integer number of resources in seconds that algorithm is allowed to run for

           output: Pandas DataFrame containing optimal configuration and corresponding
                    final farm wake loss.
            '''

        print("Successive Halving started \n")
        print("Budget: %s seconds" % budget, "\n")
        print("Number of arms/configurations: %s" % num_configurations)

        seed = 999

        # asserting that parameters are within allowed bounds
        assert False not in (x in ['a', 'b', 'c', 's', 'm'] for x in method), \
            "Wrong parameter input. The only possible methods are 'a', 'b', 'c', 's' and 'm'."

        assert multiplier[0] <= multiplier[1] and len(multiplier) == 2, \
            "Wrong parameter input. There should be 2 float or integer bounds in multiplier, where \
            the upper bound has to be greater than or equal to the lower bound"

        assert scaling[0] <= scaling[1] and scaling[2] <= scaling[3] and len(scaling) == 4, "Wrong parameter input. \
                    There should be 4 float or integer numbers in scaling: \
                    a lower and upper bound for the lower scaling bound and a lower and upper bound \
                    for the upper scaling bound, where the upper bounds are greater than or equal to \
                    its respective lower bounds"

        resource_unit = 4

        all_configurations = {}

        # creating uniformely sampled configurations
        for conf in range(num_configurations):

            param_method = method
            param_multiplier = random.uniform(multiplier[0], multiplier[1])
            scaling_lower_bound = random.uniform(scaling[0], scaling[1])
            if scaling_lower_bound > scaling[2]:
                scaling_upper_bound = random.uniform(scaling[2], scaling[3])
            else:
                scaling_upper_bound = random.uniform(scaling_lower_bound, scaling[3])
            param_scaling = [scaling_lower_bound, scaling_upper_bound]

            all_configurations[conf + 1] = [param_method, param_multiplier, param_scaling]

        # successive halving
        # initializations
        S_begin = list(all_configurations.values())

        S_dict = {0: S_begin}
        r_dict = {}
        loss_dict = {}
        R = 0

        # create configurations' models
        models = helper_functions.build_models(S_dict[0], self.wflop_problem)

        k_final = math.ceil(math.log(num_configurations, 2))

        start_time = time.time()

        # run all models for currently allowed resources
        for k in range(k_final):

            r = math.floor(budget / (len(S_dict[k]) * k_final))
            r_dict[k] = r

            R += r

            print("All arms in S%s: " % k, S_dict[k], "\n")
            print("Running optimization for all arms in S%s for exactly r(k) = %s seconds" % (k, r))

            objectives_dict = {}
            rep_values = {}

            objectives_dict = helper_functions.loop_optimizer(S_dict[k], models, r_dict[k])

            # throw out bad performing half of models
            S_dict[k + 1] = helper_functions.filter_top_k_configs(objectives_dict, 2)

            # storing best objective value
            if k == (k_final - 1):
                best_obj_val = objectives_dict[repr(S_dict[k][0])]['Objective value']

        end_time = time.time()
        budget_used = end_time - start_time

        print("From the budget of %s seconds, %s seconds were used \n" % (budget, budget_used))
        print("The last remaining hyperparameter configuration is ", S_dict[k_final - 1][0], "\n")
        print("This configuration gives an objective value of %s" % best_obj_val)

        # creating dataframe for all results
        result_storage = [
            [self.wflop_problem, S_dict[k_final - 1][0][1], S_dict[k_final - 1][0][2], S_dict[k_final - 1][0][0], '-', \
             best_obj_val, budget_used]]
        results = pd.DataFrame(result_storage,
                               columns=['File name', 'Multiplier', 'Scaling', 'Method', \
                                        'Max. Iterations', 'Objective Value Run', 'Runtime In Seconds'])

        return results

    def hyperband(self, method=['abcs'], multiplier=[1.0], scaling=[[0.8,1.2]], resources_per_config=10, eta=3):
        '''Performs hyperband using preset resources per configuration and aggressiveness parameter eta;

            input:-self.wflop_problem: (string) file path
                  -method: list of at least one and at most all combinations of the strings in {'s','a','b','c','m'};
                  -multiplier: range [a,b], where 0<=a<=b;
                  -scaling: ranges [a,b,c,d]; where 0<=a<=b<=c<=d; a and b create the range for the lower scaling bound
                   and c and d for the upper scaling bound;
                  -resources_per_config: integer number for maximum amount of resources
                   allowed per configuration in seconds
                  -eta: integer number for eta, if unsure use eta = 3

           output: Pandas DataFrame containing optimal configuration and corresponding
                    final farm wake loss.
            '''

        print("Hyperband started with R = %s and eta = %s \n" % (resources_per_config, eta))

        # hyperband
        # initializations
        s_max = math.floor(math.log(resources_per_config, eta))
        big_b = (s_max + 1) * resources_per_config
        bracket_results = {}

        total_time_start = time.time()

        # iterating over brackets
        for s in range(s_max, -1, -1):
            start_bracket = time.time()
            print("Bracket %s started \n" % s)

            # calculating currently allowed resources
            numb_configs = math.ceil((big_b / resources_per_config) * ((eta ** s) / (s + 1)))
            resources_per_round = resources_per_config / (eta ** s)

            # creating uniformely sampled configs and models
            configurations = helper_functions.create_uniform_configurations(numb_configs, method=method,
                                                                            multiplier=multiplier, scaling=scaling)
            configurations = list(configurations.values())
            models = helper_functions.build_models(list_of_configurations=configurations,
                                                   problem_file=self.wflop_problem)

            # running successive halving for current bracket parameters
            for i in range(s + 1):
                current_n = math.floor(numb_configs / (eta ** i))
                current_r = resources_per_round * (eta ** i)

                print("Sucessive halving started, round %s within bracket %s \n" % (i, s))
                print("Number of arms/configurations: %s \n" % current_n)
                print("All arms in current iteration: %s \n" % configurations)
                print(
                    "Running optimization for all arms in current iteration for exactly r(k) = %s seconds" % current_r)

                objectives_dict = helper_functions.loop_optimizer(list_of_configurations=configurations,
                                                                  dictionary_of_models=models, r=current_r)

                best_k_num = math.floor(current_n / eta)
                configurations = helper_functions.filter_abs_k_configs(objectives_dict, best_k_num)

                if i == s and s != 0:
                    bracket_results[s] = objectives_dict[repr(configurations[0])]

                elif i == s and s == 0:
                    bracket_results[s] = objectives_dict[repr(configurations[0])]

            bracket_time = time.time() - start_bracket
            print("Bracket %s took %s seconds to complete" % (s, bracket_time))

        total_time = time.time() - total_time_start
        print("Hyperband took %s seconds to complete" % total_time)

        # print("The last remaining hyperparameter configuration is ", S_dict[k_final-1][0], "\n")
        # print("This configuration gives an objective value of %s" % best_obj_val)

        # storing best objective value
        min = 100000
        for i in bracket_results.values():
            if i['Objective value'] <= min:
                min = i['Objective value']
                best = i

        # creating df for all results
        result_storage = [
            [self.wflop_problem, best['Configuration'][1], best['Configuration'][2], best['Configuration'][0], '-',
             best['Objective value'], total_time]]
        results = pd.DataFrame(result_storage,
                               columns=['File name', 'Multiplier', 'Scaling', 'Method', 'Max. Iterations',
                                        'Objective Value Run', 'Runtime In Seconds'])

        return results

    def hyperUCB(self, method=['abcs'], multiplier=[1.0], scaling=[[0.8,1.2]],
                 resources_per_config=10, eta=3, alpha=0.3, gamma=0.1 ):
        '''Performs hyperband using preset resources per configuration, aggressiveness parameter eta,
             exploration trade-off parameter alpha and regularization parameter gamma;

            input:-self.wflop_problem: (string) file path
                  -method: list of at least one and at most all combinations of the strings in {'s','a','b','c','m'};
                  -multiplier: range [a,b], where 0<=a<=b;
                  -scaling: ranges [a,b,c,d]; where 0<=a<=b<=c<=d; a and b create the range for the lower scaling bound
                   and c and d for the upper scaling bound;
                  -resources_per_config: integer number for maximum amount of resources
                   allowed per configuration in seconds
                  -eta: integer number for eta, if unsure use eta = 3
                  -alpha: float
                  -gamma: float

           output: Pandas DataFrame containing optimal configuration and corresponding
                    final farm wake loss.
            '''

        print("HyperUCB started with R = %s and eta = %s \n" % (resources_per_config, eta))
        total_time_start = time.time()

        # hyperband initialization
        s_max = math.floor(math.log(resources_per_config, eta))
        big_b = (s_max + 1) * resources_per_config
        bracket_results = {}

        # UCB initialization

        d = 3

        zero_vector = np.zeros((d, 1))
        identity_matrix = gamma * np.identity(d)
        n_0 = eta ** s_max

        for s in range(s_max, -1, -1):
            start_bracket = time.time()
            print("\nBracket %s started \n" % s)
            # calculate current hyperband parameters
            numb_configs = math.ceil((big_b / resources_per_config) * ((eta ** s) / (s + 1)))
            resources_per_round = resources_per_config / (eta ** s)

            # create current configurations and models according to top_ucb proportion
            new_configurations = helper_functions.create_uniform_configurations(n=n_0, method=method,
                                                                                multiplier=multiplier,
                                                                                scaling=scaling, ucb=True)
            models = helper_functions.build_models(list_of_configurations=list(new_configurations.values()),
                                                   problem_file=self.wflop_problem)
            current_big_lambda = helper_functions.top_ucb(new_configurations, zero_vector,
                                                          identity_matrix, numb_configs, alpha)

            # update configuration matrix and initialize score vector
            for key, value in current_big_lambda.items():
                if key == 0:
                    big_x = np.array([value])
                else:
                    big_x = np.append(big_x, [value], axis=0)

            for i in range(s + 1):
                # calculate inner loop's n and r parameters for successive halving
                current_n = math.floor(numb_configs / (eta ** i))
                current_r = resources_per_round * (eta ** i)

                # updating config matrix
                for value in current_big_lambda.values():
                    config_vector = np.array([value])
                    addition = np.matmul(config_vector, config_vector.reshape(len(config_vector[0]), 1))
                    identity_matrix = identity_matrix + addition

                print("Sucessive halving started, round %s within bracket %s \n" % (i, s))

                # running optimizer for current configs
                objectives_dict = helper_functions.loop_optimizer(list(current_big_lambda.values()),
                                                                  models, current_r, ucb=True)

                print("Number of arms/configurations: %s \n" % current_n)
                print("All arms in current iteration: %s \n" % list(new_configurations.values()))
                print(
                    "Running optimization for all arms in current iteration for exactly r(k) = %s seconds" % current_r)
                print('Objectives dict: ', objectives_dict, '\n')

                y_vector_list = [i['Objective value'] for i in objectives_dict.values()]

                start = True

                # updating score vector
                for y in y_vector_list:
                    if start:
                        y_vector = np.array([[y]])
                        start = False
                    else:
                        y_vector = np.append(y_vector, [[y]], axis=0)

                product = np.matmul(big_x.T, big_x) + gamma
                zero_vector = np.matmul(np.linalg.inv(product), big_x.T)
                zero_vector = np.matmul(zero_vector, y_vector)

                # filtering configurations based on current throw-out proportion
                current_big_lambda = helper_functions.top_ucb(current_big_lambda, zero_vector, identity_matrix,
                                             math.floor(current_n / eta), alpha)

                for key, value in current_big_lambda.items():
                    if key == 0:
                        big_x = np.array([value])
                    else:
                        big_x = np.append(big_x, [value], axis=0)

                        # storing results
                if i == s and s != 0:
                    bracket_results[s] = objectives_dict[repr(list(current_big_lambda.values())[0])]

                elif i == s and s == 0:
                    bracket_results[s] = objectives_dict[repr(list(current_big_lambda.values())[0])]

            bracket_time = time.time() - start_bracket
            print("Bracket %s took %s seconds to complete" % (s, bracket_time))

        total_time = time.time() - total_time_start
        print("HyperUCB took %s seconds to complete" % total_time)

        # stroring best objective value
        min = 100000
        for i in bracket_results.values():
            if i['Objective value'] <= min:
                min = i['Objective value']
                best = i

        # creating df for all results
        result_storage = [
            [self.wflop_problem, best['Configuration'][0], [best['Configuration'][1],
                                    best['Configuration'][2]], 'abcs', '-', best['Objective value'], total_time]]

        results = pd.DataFrame(result_storage,
                               columns=['File name', 'Multiplier', 'Scaling', 'Method', 'Max. Iterations',
                                        'Objective Value Run', 'Runtime In Seconds'])

        return results


e1 = Experiment("C:\\Users\\20175862\\Documents\\BEP\\wflopg code\\documents\\IEA37_CS1+2")
e1.fast_run(max_iterations=10)
