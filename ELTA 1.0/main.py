import logging
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from models.swin_model import Swin
from dataset import AVADataset
from utils import *
from mixup import mixup
from metric import get_metrics
import args
import nni
from nni.utils import merge_parameter
import time
import os

opt = args.init()
logger = logging.getLogger('AVA_AutoML')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")

def adjust_learning_rate(params, optimizer, epoch):
    lr = params['lr'] * (1.0 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(opt,model, loader, optimizer, criterion):
    model.train()
    train_loss = AverageMeter()

    for _, (inputs, targets) in enumerate(tqdm(loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        features,_, outputs=model(inputs) 

        if opt['mixup']:
            features, targets=mixup(features,targets,opt['tau_1'],opt['tau_2'])
            if opt['loss_type'] == 'emd':
                outputs = model.softmax(model.linear_emd(features))
            else:
                outputs = torch.squeeze(model.linear_mse(features))
        loss = criterion(outputs, targets)
        if opt['simloss_weight'] > 0:
            loss += opt['simloss_weight'] * simloss(features, targets)
        loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), outputs.size(0))
    return train_loss.avg



def validate(opt,model, loader, criterion, mode='val'):
    model.eval()
    validate_loss = AverageMeter()
    labels, preds = [], []
    all_logits = pd.DataFrame()
    all_features = pd.DataFrame()
    all_outputs = pd.DataFrame()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            targets = targets.type(torch.FloatTensor)
            targets = targets.to(device)
            features,logits,outputs = model(inputs)
            loss = criterion(targets, outputs)
            validate_loss.update(loss.item(), inputs.size(0))
            all_features=pd.concat([all_features, pd.DataFrame(features.data.cpu().numpy())],
                                         ignore_index=True)
            all_logits = pd.concat([all_logits, pd.DataFrame(logits.data.cpu().numpy())],
                                         ignore_index=True)
            all_outputs = pd.concat([all_outputs, pd.DataFrame(outputs.data.cpu().numpy())],
                                         ignore_index=True)
            if opt['loss_type'] == 'emd':
                preds.extend(get_score(outputs))
                labels.extend(get_score(targets))
            elif opt['loss_type'] == 'mse':
                preds.extend(outputs.data.cpu().numpy().tolist())
                labels.extend(targets.data.cpu().numpy().tolist())

    if opt['loss_type'] == 'mse':
        preds=[pred*10 for pred in preds]
        labels=[label*10 for label in labels]

    if mode == 'test':
        output_evaluation_results(str(opt['evaluate']), preds, all_features,all_logits,all_outputs)
    plcc,srcc,mae,mse=get_metrics(preds,labels)
    print(f'plcc: {plcc:.4f}, srcc: {srcc:.4f}, mae: {mae:.4f}, mse: {mse:.4f}')
    return validate_loss.avg, plcc, srcc

def prepare_dataloader(opt):
    train_dataset = AVADataset(os.path.join(opt['csv_path'], 'train.csv'), opt['dataset_path'], opt['st'], mode='train', 
                             loss_type=opt['loss_type'])
    val_dataset = AVADataset(os.path.join(opt['csv_path'], 'val.csv'), opt['dataset_path'], opt['st'], mode='val',
                             loss_type=opt['loss_type'])
    test_dataset = AVADataset(os.path.join(opt['csv_path'], 'test.csv'), opt['dataset_path'], opt['st'], mode='test',
                              loss_type=opt['loss_type'])
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False,drop_last=True)

    return train_loader,val_loader,test_loader

def main(opt):
    metric = opt['metric']
    global best_metric
    best_metric = 0.0

    localtime = time.strftime('%m%d_%H%M', time.localtime())
    ckpt_prefix = os.path.join('checkpoint', f'{localtime}')
    print(f"checkpoint file path:{ckpt_prefix}")
    print('==================> Creating Model')

    model = Swin(loss_type=opt['loss_type'])
    model = model.to(device)

    train_loader,val_loader,test_loader=prepare_dataloader(opt)
    if opt['loss_type'] == 'mse':
        criterion = nn.MSELoss()
    elif opt['loss_type'] == 'emd':
        criterion = EMDLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=opt['lr'])


    if opt['resume']:
        if os.path.isfile(opt['resume']):
            if not opt['retrain']:
                ckpt_prefix = os.path.join('checkpoint', opt['resume'].split('/')[-2])
            print(f"===> Loading checkpoint '{opt['resume']}'")
            checkpoint = torch.load(opt['resume'], map_location=torch.device(f"cuda:{str(opt['gpu_id'])}"))
            opt['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{opt['resume']}' (epoch {checkpoint['epoch']})")
            print(f'best {metric} : {best_metric}')
        else:
            raise ValueError(f"No checkpoint found at '{opt['resume']}'")

    if opt['retrain']:
        for name, param in model.named_parameters():
            if 'linear_emd' in name or 'linear_mse' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        opt['start_epoch'] = 0

    if opt['evaluate']:
        test = AVADataset(opt['evaluate'], opt['test_dataset_path'], opt['st'], mode='test',
                          loss_type=opt['loss_type'])
        test_loader = DataLoader(test, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
        validate(opt, model=model, criterion=criterion, loader=test_loader, mode='test')
        return

    if opt['st']:
        opt['start_epoch'] = 0

    with open(f'{localtime}.csv', 'w') as f:
        f.write(f'epoch,t_lcc,t_srcc,train_loss,test_loss,best_{metric}\r\n')
        f.flush()

        for e in range(opt['start_epoch'], opt['num_epoch']):
            adjust_learning_rate(opt, optimizer, e)
            print(f"train epoch {e + 1}/{opt['num_epoch']}  (lr:{optimizer.param_groups[0]['lr']}) :")
            train_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer,criterion=criterion)
            print('train loss: %0.4f' % train_loss)
            print(f"val epoch {e + 1}/{opt['num_epoch']}:")
            val_loss, vlcc, vsrcc = validate(opt,model=model, loader=val_loader,criterion=criterion)
            print('validate loss: %0.4f' % val_loss)

            if metric == 'srcc':
                current_metric = vsrcc
            elif metric == 'lcc':
                current_metric = vlcc

            update_best = current_metric > best_metric
            if update_best:
                best_metric = current_metric
            print(f'best_{metric}:{best_metric:.4f}')

            if not os.path.exists(ckpt_prefix):
                os.makedirs(ckpt_prefix)
            save_info = {
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_metric
            }
            torch.save(save_info, os.path.join(ckpt_prefix, 'ckpt.pth'))
            if update_best:
                # torch.save(save_info, os.path.join(ckpt_prefix, 'best.ckpt.pth'))
                torch.save(save_info, os.path.join(ckpt_prefix, f'epoch{e}_srcc_{vsrcc}.ckpt.pth'))

            f.write(
                '%d,%.4f,%.4f,%.4f,%.4f,%.4f\r\n'
                % (e + 1, vlcc, vsrcc, train_loss, val_loss, best_metric))
            f.flush()

            nni.report_intermediate_result(
                {'default': vsrcc, "vlcc": vlcc, "val_loss": val_loss})

    nni.report_final_result({'default': vsrcc, "vlcc": vlcc})


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    try:
        tuner_params = nni.get_next_parameter()
        print(tuner_params)
        logger.debug(tuner_params)
        params = vars(merge_parameter(opt, tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)

        ## nohup python train.py > stdout.log 2> stderr.log &

