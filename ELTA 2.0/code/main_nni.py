from nni.experiment import Experiment

# 创建实验对象
# experiment = Experiment('local')

# 设置搜索空间
search_space = {
    'batch_size': {'_type': 'choice', '_value': [16,32]},
    'lr': {'_type': 'choice', '_value': [1e-5,1e-4]},
    # 'simloss_weight': {'_type': 'choice', '_value': [0.01,0.1,0.2,0.5,1.0,5,10,100]},
    'tau_1': {'_type': 'choice', '_value': [0.05,0.1,0.2,0.3,0.4,0.5]},
    'tau_2': {'_type': 'choice', '_value': [2,3,4,5,6,7,10]},
}

experiment = Experiment('local')
experiment.config.trial_command = 'python main.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 8
experiment.config.trial_concurrency = 1
experiment.run(8089)
input()

# experiment.config.search_space = search_space

# # 设置 Trial 配置
# experiment.config.trial_command = 'python main.py'
# experiment.config.trial_code_directory = '.'
# experiment.config.trial_concurrency = 1
# experiment.config.max_trial_number = 8

# # 设置 Tuner
# experiment.config.tuner.name = 'TPE'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'maximize'
# }

# # 设置实验基本信息
# experiment.config.experiment_name = 'my_experiment'

# # 启动实验
# try:
#     experiment.run(port=8082)
# except Exception as e:
#     print(f"Experiment failed to start: {e}")


# input("Press Enter to stop...")
