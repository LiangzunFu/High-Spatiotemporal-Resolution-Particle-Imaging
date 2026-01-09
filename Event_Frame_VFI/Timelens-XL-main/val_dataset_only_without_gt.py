import argparse
import os
from torchvision.transforms import ToTensor, ToPILImage
import torch
import losses
from torch.utils.data import DataLoader
from tools.model_deparse import deparse_model, save_model
from tools.initOptimScheduler import init_optimizer
from tqdm import tqdm
from tools.random_seed import set_random_seed
from params.train_params import traintest_RC_smallmix_lpips
from dataset.mixloader import MixLoader_without_gt
from model.FinalBidirectionAttenFusion import FinalBidirectionAttenfusion

toim = ToPILImage()

mkdir = lambda x:os.makedirs(x, exist_ok=True)

def main(args):
    print('In Network')
    print(f"interp_ratio: {args.val_interp_ratio}")
    # save_path = rf"E:\StudyMaterials\论文 专利 项目书等\论文\Event-Frame-VFI\Event VFI results\large model with enhancedloader new mask 1 0.1\ResOut_Event_Frame_x{args.val_interp_ratio}_{args.specified_folder}"
    save_path = rf"./ResOut_Event_Frame_x{args.val_interp_ratio}_{args.specified_folder}"
    args.save_path = save_path
    # params config
    params = traintest_RC_smallmix_lpips(args)
    set_random_seed(params.args.seed)
    # define model
    # model: Expv8_large->FinalBidirectionAttenfusion  epoch, metric:{}
    model, epoch, metrics, scheduler_dict = deparse_model(params)

    # define loader
    # define testing dataloader

    test_num_workers = 4 if 'num_workers' not in params.validation_config.keys() else params.validation_config.num_workers
    testDataset = MixLoader_without_gt(dataset_path=params.args.dataset_path, training=False,
                            verify_the_specified_folder = args.specified_folder,
                            echannel=params.model_config.define_model.echannel,
                            test_interp_ratio=1,
                            train_split_ratio=params.training_config.train_split_ratio)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=test_num_workers)
    print('[Testing Samples Num]', len(testDataset))
    print("Num workers", test_num_workers)
    # define training dataloader

    print('Do not train, start validation')

    cepoch = 0
    torch.cuda.empty_cache()

    for _, testdata in tqdm(enumerate(testLoader), ascii=True):
        with torch.no_grad():
            left_frame, right_frame, events = testdata['im0'].cuda(), \
                testdata['im1'].cuda(), testdata['events'].cuda()

            interp_ratio = args.val_interp_ratio
            real_interp = interp_ratio if params.real_interp is None else params.real_interp
            jump_ratio = interp_ratio // real_interp
            end_tlist = range(jump_ratio - 1, interp_ratio - 1, jump_ratio)

            res = model.net(torch.cat((left_frame, right_frame), 1), events, interp_ratio, end_tlist)
            recon = res[0]

            rgb_name = testdata['rgb_name']
            folder = testdata['folder'][0]
            # 左一帧GT
            toim(left_frame[0].detach().cpu().clamp(0, 1)).save(
                os.path.join(save_path, f"{folder}_{rgb_name[0][0]}_gt_{0}.jpg"))
            # 中间生成帧
            for n in range(recon.shape[1]):
                os.makedirs(save_path, exist_ok=True)

                toim(recon[0, n].detach().cpu().clamp(0, 1)).save(
                    os.path.join(save_path, f"{folder}_{rgb_name[0][0]}_res_{n+1}.jpg"))


    if not params.enable_training:
        print('Training is not enabled...Validation finished, end the process...')
        # exit()

    # return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, default=r"D:\dataset\Frame_Event_Dataset")
    parser.add_argument('--dataset_path', type=str, default=r"/public/home/u41731/dataset/Frame_Event_Dataset/")
    # parser.add_argument('--dataset_path', type=str, default=r"D:\dataset\Frame_Event_Dataset")
    parser.add_argument('--specified_folder', type=str, default="dataset_15")
    parser.add_argument('--val_interp_ratio', type=int, default=16)  # 2,4,8,16
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--param_name", type=str, default="traintest_RC_smallmix_lpips", help="model saving path")
    parser.add_argument("--model_name", type=str, default="Expv8_large", help="model name")
    # parser.add_argument("--model_pretrained", type=str, default="/mnt/fuliangzun/python_project/TimeLens-XL-main/Expv8_large_HQEVFI.pt", help="model saving name")
    parser.add_argument("--model_pretrained", type=str,
                        default='/public/home/u41731/pythonproject/Event_Frame_VFI/new_mask_1_0.1.pt', help="model saving name")
    # parser.add_argument("--model_pretrained", type=str, default=r"E:\StudyMaterials\pythonproject\Event_Frame_VFI\large_enhanced_mask0.01.pt", help="model saving name")
    # parser.add_argument("--init_step", type=int, default=None, help="initialize training steps")
    parser.add_argument("--skip_training", default=True, help="Whether or not enable training")
    parser.add_argument("--clear_previous", default=True, help="Delete previous results")
    parser.add_argument("--extension", type=str, default='', help="extension of save folder")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--calc_flops", default=False)
    parser.add_argument("--debug", default=False)
    parser.add_argument("--save_flow", type=bool, default=False, help="save optical flow or not")
    parser.add_argument("--save_images", type=bool, default=True, help="save images for validation or not")
    args = parser.parse_args()
    for i in [2,4,8,16]:
        args.val_interp_ratio = i
        main(args)
