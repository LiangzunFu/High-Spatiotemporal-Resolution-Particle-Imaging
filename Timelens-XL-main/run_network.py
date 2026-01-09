import argparse
import time

import torch
import losses
from torch.utils.data import DataLoader
from tools.model_deparse import deparse_model, save_model
from tools.initOptimScheduler import init_optimizer
from tqdm import tqdm
from tools.random_seed import set_random_seed
from params.train_params import traintest_RC_smallmix_lpips
from dataset.mixloader import BaseMixLoader, EnhancedMixLoader, EnhancedMixLoaderWithMask

def main(args):
    print('In Network')
    # params config
    params = traintest_RC_smallmix_lpips(args)

    set_random_seed(params.args.seed)
    # define model
    # model: Expv8_large->FinalBidirectionAttenfusion  epoch, metric:{}
    model, epoch, metrics, scheduler_dict = deparse_model(params)

    # define loader
    # define testing dataloader
    test_num_workers = 4 if 'num_workers' not in params.validation_config.keys() else params.validation_config.num_workers
    testDataset = EnhancedMixLoaderWithMask(params, training=False, verify_the_specified_folder = None)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=test_num_workers)
    print('[Testing Samples Num]', len(testDataset))
    print("Num workers", test_num_workers)
    # define training dataloader
    if params.enable_training:
        trainDataset = EnhancedMixLoaderWithMask(params, training=True)
        trainLoader = DataLoader(trainDataset, batch_size=params.training_config.batch_size, shuffle=True,
                                 num_workers=params.training_config.num_workers, drop_last=True,
                                 pin_memory=True, prefetch_factor=5)
        print('[Training Samples Num]', len(trainDataset))
        params.training_config.optim.scheduler_params.T_max = len(trainDataset) * params.training_config.max_epoch
        # init optimizer and scheduler
        optimizer, scheduler, scheduler_type, epoch = init_optimizer(model, epoch, params, scheduler_dict)
    for cepoch in range(epoch, params.training_config.max_epoch):
        model._update_training_time()
        if params.enable_training:
            # st = time.time()
            for step, data in enumerate(trainLoader):
               #print(step, time.time()-st)
               model.net_training(data, optimizer, cepoch, step)
               if scheduler is not None and scheduler_type == 'step':
                   scheduler.step()
               # time.time()
            if scheduler is not None and scheduler_type == 'epoch':
                scheduler.step()
            log_cont = model._print_train_log(cepoch)
            save_model(cepoch, metrics, params, model, scheduler)
        else:
            print('Do not train, start validation')
            log_cont = f'Only for Validation EPOCH: {cepoch} '
            params.validation_config.val_epochs = cepoch
            params.validation_config.val_imsave_epochs = cepoch

        if (cepoch % max(params.validation_config.val_epochs, 1) == 0 and cepoch > 0) or not params.enable_training:
            print(f"val_interp_ratio:{args.val_interp_ratio}")
            if not params.enable_training:
                print(f"pretrained model weight:{args.model_pretrained}")
            torch.cuda.empty_cache()
            for _, testdata in tqdm(enumerate(testLoader)):
                model.net_validation(testdata, cepoch)
            log_cont = log_cont.strip('\n')+'\t'
            log_cont += model._print_val_log()
            log_cont = log_cont.strip('\t')+'\n'
        model._init_metrics(None)
        model.write_log(log_cont)
        if not params.enable_training:
            print('Training is not enabled...Validation finished, end the process...')
            exit()

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 本地数据集地址
    # parser.add_argument('--dataset_path', type=str, default=r"D:\dataset\Frame_Event_Dataset")
    # 实验室服务器数据集地址
    # parser.add_argument('--dataset_path', type=str, default=r"/mnt/fuliangzun/dataset/Frame_Event_Dataset/")
    # 超算数据集地址
    parser.add_argument('--dataset_path', type=str, default=r"/public/home/u41731/dataset/Frame_Event_Dataset/")
    parser.add_argument('--save_path', type=str, default=r"./ResOut_Event_Frame")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--param_name", type=str, default="traintest_RC_smallmix_lpips", help="model saving path")
    parser.add_argument("--model_name", type=str, default="Expv8_large", help="model name")
    # parser.add_argument("--model_pretrained", type=str, default="/mnt/fuliangzun/python_project/TimeLens-XL-main/Expv8_large_HQEVFI.pt", help="model saving name")
    parser.add_argument("--model_pretrained", type=str, default=None, help="model saving name")
    # parser.add_argument("--model_pretrained", type=str, default='/public/home/u41731/pythonproject/Event_Frame_VFI/large.pt', help="model saving name")
    parser.add_argument("--skip_training", default=True, help="Whether or not enable training")
    parser.add_argument("--val_interp_ratio", type=int,default=8, help="The validation interpolation ratio")
    parser.add_argument("--clear_previous", default=False, help="Delete previous results")
    parser.add_argument("--extension", type=str, default='', help="extension of save folder")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--calc_flops", default=False)
    parser.add_argument("--debug", default=False)
    parser.add_argument("--save_flow", type=bool, default=False, help="save optical flow or not")
    parser.add_argument("--save_images", type=bool, default=True, help="save images for validation or not")
    args = parser.parse_args()

    main(args)
