search_space = {
    'batch_size': {'_type': 'choice', '_value': [16,48]},
    'lr': {'_type': 'choice', '_value': [1e-5,1e-4]},
    'simloss_weight': {'_type': 'choice', '_value': [0.01,0.1,0.2,0.5,1.0,5,10,100]},
    'tau_1': {'_type': 'choice', '_value': [0.05,0.1,0.2,0.3,0.4,0.5]},
    'tau_2': {'_type': 'choice', '_value': [2,3,4,5,6,7,10]},
}

from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.trial_command = 'python main.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 8
experiment.config.trial_concurrency = 1
experiment.run(8082)
input()
