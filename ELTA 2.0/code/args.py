import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    # path
    parser.add_argument('--csv_path', type=str, default="/home/llm/elta/code/dataset_backup/aadb",
                        help='path to dataset-label csv file')
    parser.add_argument('--dataset_path', type=str, default='/data/dataset/aadb',
                        help='path to dataset')
    parser.add_argument('--test_dataset_path', type=str, default='/data/dataset/aadb',
                        help='path to test_dataset')

    # params
    parser.add_argument('--loss_type', type=str, choices=['emd', 'mse'], default='mse')
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--start_epoch', default=0, type=int, help='which epoch to start training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--metric', default='srcc', type=str, choices=['srcc', 'lcc', 'acc'],
                        help='the metric for updating checkpoint')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='the checkpoint to resume training')
    parser.add_argument('-e', '--evaluate', default=None, dest='evaluate', type=str,
                        help='evaluate and generate pseudo-labels')
    parser.add_argument('--st', default=False, action='store_true',
                        help='whether to enable self-training')
    parser.add_argument('--retrain', default=False, action='store_true',
                        help='whether to use RRT:regressor retraining')
    
    parser.add_argument('--mixup', default=True, action='store_true', help='whether to enable mixup')
    parser.add_argument('--tau_1', type=float, default=0.5)
    parser.add_argument('--tau_2', type=float, default=2)
    parser.add_argument('--simloss_weight', type=float, default=0.0)

    args = parser.parse_args()
    return args
