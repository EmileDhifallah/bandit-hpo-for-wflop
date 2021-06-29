# wflopg library
import wflopg
from wflopg.create_layout import fix_constraints
from wflopg.helpers import rss
from wflopg.optimizers import _step_generator
from wflopg.optimizers import _iterate_visualization
from wflopg.optimizers import _setup_visualization

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# basic helper libraries
import random
import math
import time
from collections import Counter

# data manipulation libraries
import numpy as np
import pandas as pd
import xarray as _xr


def create_uniform_configurations(n=10, max_iterations=[], method=[], multiplier=[], scaling=[], ucb=False):
    '''
    Creates n uniformely randomly selected hyperparameter configurations.

    Input: number of configurations desired, lists of ranges for hyperparameters
    Output: dictionary containing number range as keys and configurations as values
    '''
    seed = 999

    if max_iterations:
        assert max_iterations[0] <= max_iterations[1] and len(max_iterations) == 2, "Wrong parameter input. \
                                There should be 2 float or integer bounds in max_iterations, where \
                                the upper bound has to be greater than or equal to the lower bound"
    if method:
        assert False not in (x in ['a', 'b', 'c', 's', 'm'] for x in
                             method), "Wrong parameter input. The only possible methods are 'a', 'b', 'c', 's' and 'm'."

    if multiplier:
        assert multiplier[0] <= multiplier[1] and len(multiplier) == 2, "Wrong parameter input. There should be 2 " \
                                "float or integer bounds in multiplier, where the upper bound has to be greater \
                                than or equal to the lower bound"

    if scaling:
        assert scaling[0] <= scaling[1] and scaling[2] <= scaling[3] and len(scaling) == 4, "Wrong parameter input. \
                                There should be 4 float or integer numbers in scaling: \
                                a lower and upper bound for the lower scaling bound and a lower and upper bound \
                                for the upper scaling bound, where the upper bounds are greater than or equal to \
                                its respective lower bounds"

    all_configurations = {}

    for conf in range(n):
        conf_list = []
        if max_iterations:
            param_max_iterations = math.ceil(random.uniform(max_iterations[0], max_iterations[1]))
            conf_list.append(param_max_iterations)

        if method:
            method_string = ''.join(method)
            num_methods = random.randint(1, len(method_string))
            param_method = ''.join(random.sample(method_string, num_methods))
            conf_list.append(param_method)

        if multiplier:
            param_multiplier = random.uniform(multiplier[0], multiplier[1])
            conf_list.append(param_multiplier)

        if scaling:
            scaling_lower_bound = random.uniform(scaling[0], scaling[1])
            if scaling_lower_bound > scaling[2]:
                scaling_upper_bound = random.uniform(scaling[2], scaling[3])
            else:
                scaling_upper_bound = random.uniform(scaling_lower_bound, scaling[3])
            param_scaling = [scaling_lower_bound, scaling_upper_bound]

            if ucb == True:
                conf_list.append(scaling_lower_bound)
                conf_list.append(scaling_upper_bound)
            else:
                conf_list.append(param_scaling)

            all_configurations[conf + 1] = conf_list

            return all_configurations


### updated_step_iterator() is a function that was copied from the wflopg API but altered to fit
### within the hyper_ucb() function
def updated_step_iterator(owflop, max_iterations=100,
                          methods='abc', multiplier=1, scaling=[.8, 1.1],
                          wake_spreading=False, visualize=False, time_allowed=10):
    """Iteratively optimize a layout using pseudo-gradients.
    Parameters
    ----------
        owflop
            `wflopg.Owflop` wind farm layout optimization object
        max_iterations : int
            The maximum number of iterations.
        methods : str
            String of pseudo-gradient variants to use.
            Each is identified by a single character: `'s'` for simple,
            `'a'` for push-away, `'b'` for push-back, `'c'` for push-cross, and
            `'m'` for a uniform mixture of push-away and push-back.
        multiplier : float
            Starting step multiplier.
        scaling
            List of scaling values (floats) or `'False'` for no scaling.
        wake_spreading: bool
            Whether to apply a wake spreading heuristic or not.
        visualize : bool
            Whether to visualize the optimization run or not.
    Returns
    -------
    xarray.Dataset
        The optimization run history
    """
    methods = list(methods)

    if scaling is False:
        scaling = [1]
    scaler = (
            _xr.DataArray(np.float64(scaling), dims=('scale',))
            * _xr.DataArray(np.ones(len(methods)), coords=[('method', methods)])
    )

    initial_multiplier = multiplier
    if wake_spreading:
        spread_multiplier = initial_multiplier
    else:
        spread_multiplier = 0

    def spread_factor(spread_multiplier):
        return 1 + 3 * (spread_multiplier - 0) / (initial_multiplier - 0)

    # prepare history
    iterations = _xr.DataArray(
        data=np.ones(max_iterations + 1),
        coords=[('iteration', np.arange(max_iterations + 1))]
    )
    history = _xr.Dataset(data_vars={
        'layout': _xr.full_like(owflop._ds.layout, np.nan) * iterations,
        'objective': _xr.full_like(iterations, np.nan),
        'objective_bound': _xr.full_like(iterations, np.nan),
        'max_step': _xr.full_like(iterations, np.nan),
        'actual_step': _xr.full_like(iterations, np.nan),
        'spread': _xr.full_like(iterations, np.nan),
        'corrections': _xr.full_like(iterations, "", dtype=object),
        'method': _xr.full_like(iterations, ' ', dtype=str)
    })

    owflop.calculate_deficit(spread_factor(spread_multiplier))
    owflop.calculate_power()

    # initialize history
    selector = dict(iteration=0)
    history.layout[selector] = owflop._ds.layout
    history.objective[selector] = owflop.objective()
    best = start = history.isel(iteration=0).objective
    if visualize:
        axes = _setup_visualization(owflop, history)

    total_time_spent = 0  ###edit
    iternum = 0  # number of rounds that have been completed    ###edit
    mean_iteration_time = 0  ###edit
    for k in range(1, max_iterations + 1):
        if total_time_spent + mean_iteration_time > time_allowed:  ###edit
            break  ###edit
        else:  ###edit
            start_iteration = time.time()  ###edit
            print(k, end=': ')
            owflop.process_layout(history.isel(iteration=k - 1).layout)
            spread = spread_factor(spread_multiplier)
            owflop.calculate_deficit(spread)
            # calculate step
            owflop.calculate_relative_wake_loss_vector()
            step = _xr.concat(
                [_step_generator(owflop, method) for method in methods], 'method')
            # remove any global shift
            step -= step.mean(dim='target')
            # normalize the step to the largest pseudo-gradient
            distance = rss(step, dim='xy')
            step /= distance.max('target')
            del distance
            # take step
            multiplier = scaler * multiplier
            step = step * multiplier * owflop.rotor_diameter_adim
            owflop.process_layout(owflop._ds.layout + step)
            del step
            # fix any constraint violation
            corrections = fix_constraints(owflop)
            # evaluate new layout
            owflop._ds = owflop._ds.chunk({'scale': 1, 'method': 1})
            owflop.calculate_deficit(spread)
            owflop.calculate_power()
            owflop._ds.load()
            i = owflop.objective().argmin('scale')
            owflop._ds = owflop._ds.isel(scale=i, drop=True)
            j = owflop.objective().argmin('method').item()
            owflop._ds = owflop._ds.isel(method=j, drop=True)
            layout = owflop._ds.layout
            current = owflop.objective()
            print(f"(wfl: {np.round(current.item() * 100, 4)};",
                  f"spread: {np.round(spread, 2)})", sep=' ')
            bound = best + (start - best) / k
            multiplier = multiplier.isel(scale=i, drop=True)
            current_multiplier = multiplier.isel(method=j, drop=True).item()
            max_distance = (
                    rss(layout - history.isel(iteration=k - 1).layout, dim='xy').max()
                    / owflop.rotor_diameter_adim
            ).item()
            # update history
            selector = dict(iteration=k)
            history.layout[selector] = layout
            history.objective[selector] = current
            history.objective_bound[selector] = bound
            history.max_step[selector] = current_multiplier
            history.actual_step[selector] = max_distance
            history.spread[selector] = spread_multiplier
            history.corrections[selector] = corrections
            history.method[selector] = methods[j]
            # visualization
            if visualize:
                _iterate_visualization(
                    axes, owflop, history.isel(iteration=slice(0, k + 1)))
            # check best layout and criteria for early termination
            if current < best:
                best = current
            elif current > bound:
                return history.isel(iteration=slice(0, k + 1))
            if wake_spreading:
                if current_multiplier < spread_multiplier:
                    weight = np.log2(k + 1)
                    spread_multiplier = (
                            (spread_multiplier * weight + current_multiplier)
                            / (weight + 1)
                    )
                else:
                    # force reduction of spread multiplier to avoid getting stuck
                    spread_multiplier /= 10 ** 0.01

            iternum += 1  ###edit
            iteration_time = time.time() - start_iteration  ###edit

            total_time_spent += iteration_time  ###edit

            mean_iteration_time = mean_iteration_time + ((iteration_time - mean_iteration_time) / iternum)  ###edit

    return history

def loop_optimizer(list_of_configurations, dictionary_of_models, r, ucb=False):
    '''Runs step iterator for all models in input;
    Input: -list of lists with configurations;
           -dictionary of models with the shape {repr(configuration_1): model_1, ...};
            resources per optimization run;
    Output: dictionary of objective value per optimization run with the shape
            {repr(configuration_1): {'Objective value': objective value_1, 'Configuration': configuration_1}, ... ...}
    '''

    objectives = {}

    for i in list_of_configurations:
        if ucb == True:
            methods = 'abc'
            multiplier = i[0]
            scaling = [i[1], i[2]]
        else:
            methods = i[0]
            multiplier = i[1]
            scaling = i[2]

        optimizer = updated_step_iterator(dictionary_of_models[repr(i)], max_iterations=100000, methods=methods,
                                          multiplier=multiplier, scaling=scaling, visualize=False, time_allowed=r)

        obj_value_l = optimizer['objective'].values
        obj_value = obj_value_l[~np.isnan(obj_value_l)][-1]
        objectives[repr(i)] = {'Objective value': obj_value, 'Configuration': i}

    return objectives


def build_models(list_of_configurations, problem_file):
    '''Builds wflopg models based on problem file for every configuration given in input;
        input: list of configurations;
        output: dictionary with configurations as keys and models as values.
    '''
    models = {}

    for model in list_of_configurations:
        models[repr(model)] = wflopg.Owflop()
        models[repr(model)].load_problem(problem_file)
        models[repr(model)].calculate_wakeless_power()

    return models


def filter_top_k_configs(objectives_dict, k):
    '''Filters the top performing configurations based on a desired throw-out proportion;

    Input: -dictionary of objective value per optimization run with the shape
            {repr(configuration_1): {'Objective value': objective value_1, 'Configuration': configuration_1}, ... ...};
           -k: float or integer: proportion of how many configurations should be thrown out
            (k=2 means that configurations will be halved);
    Output: list of lists containing the best performing configurations.
    '''

    objective_values = [i['Objective value'] for i in objectives_dict.values()]
    configurations = [i['Configuration'] for i in objectives_dict.values()]

    new_list_length = math.floor(len(objectives_dict.keys()) / k)
    print('objectives_dict', objectives_dict)
    print('k', k)
    print('new_list_length', new_list_length)
    cutoff_element = sorted(objective_values)[new_list_length]

    new_list = []

    combinations = zip(objective_values, configurations)

    for obj, conf in combinations:
        if obj < cutoff_element:
            new_list.append(conf)

    return new_list


def filter_abs_k_configs(objectives_dict, k):
    '''Filters the top k performing configurations;

    Input: -dictionary of objective value per optimization run with the shape
            {repr(configuration_1): {'Objective value': objective value_1, 'Configuration': configuration_1}, ... ...};
           -k: integer number of desired left-over configurations;
    Output: list of lists containing the best performing configurations.
    '''

    new_configurations = []

    objective_values = [i['Objective value'] for i in objectives_dict.values()]
    configurations = [i['Configuration'] for i in objectives_dict.values()]

    new_objectives_dict = {repr(configurations[i]): objective_values[i] for i in range(len(configurations))}
    count = Counter(new_objectives_dict)

    highs = count.most_common()[-k:]

    for config in highs:
        new_configurations.append(objectives_dict[config[0]]['Configuration'])

    return new_configurations


def top_ucb(current_configurations, zero_vector, identity_matrix, n, alpha):
    '''Updates configuration score vector p based on configurations' performance and exploration/exploitation trade-off;

        Input: -current_configurations: dictionary with configurations as values;
               -zero_vector: Numpy array of current score vector;
               -identity_matrix: Numpy matrix of current configuration permutations;
               -n: integer number of top performing configurations that should be left over in next iteration;
               -alpha: float number for trade-off parameter alpha;
        Output: dictionary containing new configurations subject to n.
    '''

    objectives_d = {}

    # updating score vector
    for key, value in current_configurations.items():
        config_vector = np.array(value).reshape(len(value), 1)
        config_term = np.matmul(zero_vector.T, config_vector)

        regur_vector = np.matmul(config_vector.T, np.linalg.inv(identity_matrix))
        regur_vector = np.matmul(regur_vector, config_vector)

        regur_term = alpha * math.sqrt(regur_vector)

        p = config_term + regur_term

        objectives_d[repr(value)] = {'Objective value': p, 'Configuration': value}

        # filtering configurations based on given number of configs
    new_configs = filter_abs_k_configs(objectives_d, n)
    new_range = [i for i in range(len(new_configs))]

    return dict(zip(new_range, new_configs))


def grid_search_heatmaps(grid_df, filename, multiplier_vals, scaling_vals):
    d_m = len(multiplier_vals)
    d_s = len(scaling_vals)

    data = np.reshape(np.array(grid_df['Objective Value Run']), (d_m, d_s))
    data_frame = pd.DataFrame(data, index=multiplier_vals, columns=[tuple(i) for i in scaling_vals])

    ax = sns.heatmap(data_frame, cbar_kws={'label': 'Objective Value'})
    plt.xlabel("Scaling Factor")
    plt.ylabel("Multiplier")

    filename = filename.split('.')[0]
    name = "heatmap-" + filename
    name = name.split(".")[0] + '.png'
    plt.savefig(name, bbox_inches='tight')

    plt.show()

 
