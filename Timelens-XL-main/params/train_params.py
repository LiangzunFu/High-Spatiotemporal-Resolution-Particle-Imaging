import torch
from easydict import EasyDict as ED
import os
from params.model_arch_params import model_arch_config

mkdir = lambda x:os.makedirs(x, exist_ok=True)

def traintest_RC_smallmix_lpips(args):
    paths = ED()
    paths.train_rgb = args.dataset_path
    paths.train_evs = args.dataset_path
    paths.test_rgb = args.dataset_path
    paths.test_evs = args.dataset_path

    paths.save = ED()
    paths.save.save_path = args.save_path
    paths.save.exp_path = os.path.join(paths.save.save_path,
                                       f"TimeLens-XL" + f"_adamLPIPS")
    paths.save.record_txt = os.path.join(paths.save.exp_path, 'training_record.txt')
    paths.save.train_im_path = os.path.join(paths.save.exp_path, 'training_Visual_Examples')
    paths.save.val_im_path = os.path.join(paths.save.exp_path, 'Validation_Visual_Examples')
    paths.save.weights = os.path.join(paths.save.exp_path, 'weights')

    if args.clear_previous and os.path.isdir(paths.save.exp_path):
        print(f'-- Select args.clear_previous to be True, delete previous results at {paths.save.exp_path}')
        os.system(f"rm -rf {paths.save.exp_path}")

    for k in paths.save.keys():
        if not k.endswith('_txt'):
            mkdir(paths.save[k])

    model_config = ED()
    model_config.name = 'Expv8_large'
    model_config.model_pretrained = args.model_pretrained
    cur_model_arch_config = model_arch_config[model_config.name]
    for k in cur_model_arch_config.keys():
        model_config.update({
            k: cur_model_arch_config[k]
        })

    training_config = ED()
    training_config.dataloader = 'mix_loader_smallRC'
    training_config.crop_size = 512
    # 定义裁剪范围 (left, upper, right, lower)
    training_config.crop_range = {'dataset_4':[0,0,1060,720],'dataset_6':[200,0,1120,720],
                                  'dataset_7': [170, 0, 1140, 720],'dataset_12': [470, 0, 1280, 720],
                                  'dataset_13': [770, 0, 1280, 720],'dataset_15': [0, 0, 570, 720],
                                  'dataset_16': [50, 0, 340, 720],'dataset_17': [80, 0, 360, 720],
                                  'dataset_18': [150, 0, 1280, 720],'dataset_19': [100, 0, 1240, 720],
                                  'dataset_20': [270, 0, 1280, 720],'dataset_23': [250, 0, 1280, 720],
                                  'dataset_26': [0, 0, 1100, 720],'dataset_27': [0, 0, 1050, 720]
                                  }
    training_config.test_dataset = ['dataset_10', 'dataset_14', 'dataset_16']
    # the ratio of the training dataset(train_split_ratio) and testing dataset(1-train_split_ratio), if None, the test dataset is training_config.test_dataset
    training_config.train_split_ratio = None
    # 判断是否为背景的事件数的阈值，若两张图之间的事件数量小于event_num_threshold，则最终损失乘以background_loss_weight
    training_config.event_num_threshold = 200
    training_config.background_loss_weight = 0.01

    training_config.num_workers = 16
    training_config.batch_size = 1

    training_config.data_index_offset = 0
    training_config.rgb_sampling_ratio = 1
    # training_config.sample_group = 3
    training_config.random_t = False


    # optimizer and scheduler
    training_config.optim = ED()
    training_config.optim.name = 'Adam'
    training_config.optim.optim_params = ED()
    training_config.optim.optim_params.lr = 1e-4
    # training_config.optim.name = 'Adam'
    training_config.optim.scheduler = 'CosineAnnealingLR'
    # training_config.lr = 2e-4
    training_config.optim.scheduler_params = ED()
    training_config.optim.scheduler_params.T_max = 292220
    training_config.optim.scheduler_params.eta_min = 1e-7
    # training_config.optim.scheduler = 'MultiLR'
    # training_config.lr = 1e-4
    # training_config.optim.scheduler_lr_gamma = 0.5
    # training_config.optim.scheduler_lr_milestone = [25, 50, 75]
    training_config.max_epoch = 10

    training_config.losses = ED()
    training_config.losses.Charbonier = ED()
    training_config.losses.Charbonier.weight = 1.
    training_config.losses.Charbonier.as_loss = False

    training_config.losses.MaskCharbonier = ED()
    training_config.losses.MaskCharbonier.weight = 1.
    training_config.losses.MaskCharbonier.as_loss = True

    training_config.losses.lpips = ED()
    training_config.losses.lpips.weight = 0.1
    training_config.losses.lpips.as_loss = True

    training_config.losses.psnr = ED()
    training_config.losses.psnr.weight = 1.
    training_config.losses.psnr.as_loss = False
    training_config.losses.psnr.test_y_channel = False

    # For training loss print
    training_config.train_stats = ED()
    training_config.train_stats.print_freq = 500
    training_config.train_stats.save_im_ep = 1
    # if not args.calc_flops:
    #     training_config.data_paths = parse_path_common(paths.train_rgb, paths.train_evs, RC=True)
    training_config.interp_ratio_list = 2, 4, 8, 16
    training_config.interp_list_pob = 0.25, 0.25, 0.25, 0.25

    validation_config = ED()
    validation_config.dataloader = 'mix_loader_smallRC'
    validation_config.val_epochs = 1
    validation_config.val_imsave_epochs = 2
    validation_config.weights_save_freq = 1
    validation_config.data_index_offset = 0
    validation_config.rgb_sampling_ratio = 1
    validation_config.interp_ratio = args.val_interp_ratio
    validation_config.random_t = False

    validation_config.losses = ED()
    validation_config.losses.l1_loss = ED()
    validation_config.losses.l1_loss.weight = 1.
    validation_config.losses.l1_loss.as_loss = False

    validation_config.losses.psnr = ED()
    validation_config.losses.psnr.weight = 1.
    validation_config.losses.psnr.as_loss = False
    validation_config.losses.psnr.test_y_channel = False

    validation_config.losses.maskpsnr = ED()
    validation_config.losses.maskpsnr.weight = 1.
    validation_config.losses.maskpsnr.as_loss = False

    validation_config.losses.ssim = ED()
    validation_config.losses.ssim.weight = 1.
    validation_config.losses.ssim.as_loss = False
    validation_config.losses.ssim.test_y_channel = False

    validation_config.losses.lpips = ED()
    validation_config.losses.lpips.weight = 1.
    validation_config.losses.lpips.as_loss = False

    validation_config.losses.dists = ED()
    validation_config.losses.dists.weight = 1.
    validation_config.losses.dists.as_loss = False


    params = ED()
    params.args = args
    params.paths = paths
    params.training_config = training_config
    params.validation_config = validation_config
    params.model_config = model_config
    params.real_interp = None

    params.debug = args.debug
    params.save_flow = args.save_flow
    params.save_images = args.save_images
    params.gpu_ids = range(torch.cuda.device_count())
    params.enable_training = not args.skip_training
    params.local_rank = args.local_rank

    return params
