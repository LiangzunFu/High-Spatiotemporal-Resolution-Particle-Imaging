import torch
import numpy as np
import os
import random
from numba import jit
from natsort import natsorted as sorted
from timm.data.auto_augment import brightness
from torchvision.transforms import ToPILImage, ToTensor, ColorJitter, Compose
from torch.utils.data import Dataset
from PIL import Image
import re
import random
import torch
from torchvision.transforms import ColorJitter, functional as F
from PIL import Image

class SyncColorJitter:
    """
    对多张图片同步应用相同随机参数的 ColorJitter（亮度、对比度、饱和度、色相）
    支持 Tensor 或 PIL.Image
    """
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, imgs):
        # imgs: list[Tensor/PIL.Image] 或单张 Tensor/PIL.Image
        if not isinstance(imgs, list):
            imgs = [imgs]

        # 随机生成参数
        brightness_factor = random.uniform(max(0, 1-self.brightness), 1+self.brightness) if self.brightness else 1.0
        contrast_factor   = random.uniform(max(0, 1-self.contrast), 1+self.contrast) if self.contrast else 1.0

        out_imgs = []
        for img in imgs:
            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
            img = img.clamp(0, 1)
            out_imgs.append(img)

        return out_imgs if len(out_imgs) > 1 else out_imgs[0]


@jit(nopython=True)
def sample_events_to_grid(voxel_channels, h, w, x, y, t, p):
    voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    if len(t) == 0:
        return voxel
    t_start = t[0]
    t_end = t[-1]
    t_step = (t_end - t_start + 1) / voxel_channels
    for d in range(len(x)):
        d_x, d_y, d_t = x[d], y[d], t[d]
        d_p = 1 if p[d]>0 else -1

        ind = int((d_t - t_start) // t_step)
        voxel[ind, d_y, d_x] += d_p
    return voxel

class BaseMixLoader(Dataset):
    def __init__(self, params, training=True, verify_the_specified_folder=None):
        super().__init__()
        self.params = params
        self.dataset_path = params.args.dataset_path
        self.training_flag = training
        self.crop_size = 256 if self.training_flag else None
        self.train_split_ratio = params.training_config.train_split_ratio
        self.test_dataset = params.training_config.test_dataset
        self.rgb_sampling_ratio = 1
        self.toim = ToPILImage()
        self.totensor = ToTensor()
        self.data_paths = {}
        if verify_the_specified_folder is not None and self.training_flag is False:
            selected_folders = [verify_the_specified_folder]
        else:
            # 划分数据集
            self.train_folders, self.test_folders = self.split_train_test()
            # 根据 training_flag 选择要加载的子集
            selected_folders = self.train_folders if self.training_flag else self.test_folders
        self.get_data_paths(selected_folders)
        self.samples_dict = {}
        self.interp_ratio_list = list(params.training_config.interp_ratio_list) if training else [params.validation_config.interp_ratio]
        self.interp_list_pob = list(params.training_config.interp_list_pob) if training else [1]
        # self.interp_list_pob = [1, 0, 0, 0] if training else [1]
        for irl in self.interp_ratio_list:
            self.samples_dict.update({
                str(irl):[]
            })
        self.samples_indexing()
        print(f'---- Interp List: {self.interp_ratio_list}, prob: {self.interp_list_pob}')
        self.norm_voxel = True
        self.echannel = params.model_config.define_model.echannel
        self.crop_range = params.training_config.crop_range
        self.transform = SyncColorJitter(brightness=0.2, contrast=0.2)
        self.event_num_threshold = params.training_config.event_num_threshold
        self.background_loss_weight = params.training_config.background_loss_weight

    def split_train_test(self):
        """划分训练和测试文件夹"""
        sub_folders = sorted(
            f for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f)) and re.match(r'^dataset_\d+$', f)
        )
        random.seed(self.params.args.seed)
        random.shuffle(sub_folders)
        if self.train_split_ratio is not None:
            split_index = int(len(sub_folders) * self.train_split_ratio)
            train_folders = sub_folders[:split_index]
            test_folders = sub_folders[split_index:]
        else:
            test_folders = self.test_dataset
            train_folders = [d for d in sub_folders if d not in test_folders]
        print(f"Total folders: {len(sub_folders)}, Train: {len(train_folders)}, Test: {len(test_folders)}")
        print("Train folders:", train_folders)
        print("Test folders:", test_folders)
        return train_folders, test_folders

    def get_data_paths(self, sub_folders):
        """读取指定文件夹下的所有 Frames 和 Events 文件位置"""
        for sub_folder in sub_folders:
            self.data_paths[sub_folder] = [[], []]
            self.data_paths[sub_folder][0] = [f for f in os.listdir(os.path.join(self.dataset_path,sub_folder,'Frames')) if
                              f.lower().endswith(".png")]
            self.data_paths[sub_folder][0].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][0] = [os.path.join(self.dataset_path,sub_folder,'Frames',f) for f in self.data_paths[sub_folder][0]]
            self.data_paths[sub_folder][1] = [f for f in os.listdir(os.path.join(self.dataset_path, sub_folder, 'Events')) if
                              f.lower().endswith(".npz")]
            self.data_paths[sub_folder][1].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][1] = [os.path.join(self.dataset_path, sub_folder, 'Events', f) for f in self.data_paths[sub_folder][1]]

    def samples_indexing(self):
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]

            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            for irl in self.interp_ratio_list:
                for i_ind in range(0, len(indexes) - irl, 1 if self.training_flag else irl):
                    rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + irl + 1]]
                    evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + irl]]
                    rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                    self.samples_dict[str(irl)].append([k, rgb_name, rgb_sample, evs_sample])
        return

    def __len__(self):
        return len(self.samples_dict[str(self.interp_ratio_list[0])])

    def imreader(self, impath):
        im = self.totensor(Image.open(impath).convert('RGB'))
        return im

    def events_reader(self, events_path, h, w, interp_ratio):
        evs_data = [np.load(ep) for ep in events_path]
        num_events = 0
        for ev in evs_data:
            num_events += len(ev['x'])

        loss_weight = 1
        evs_voxels = []
        for ed in evs_data:
            evs_voxels.append(
                sample_events_to_grid(self.echannel//interp_ratio, h, w, ed['x'], ed['y'], ed['t'],
                                      ed['p']))
        return torch.tensor(np.concatenate(evs_voxels, 0)), loss_weight

    def data_loading(self, paths, sample_t, interp_ratio):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events, loss_weight = self.events_reader(evs_sample, h, w, interp_ratio)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts, loss_weight

    def get_key_with_weights(self, idx):
        scalar = np.random.choice(self.interp_ratio_list, p=self.interp_list_pob)
        data_sample = self.samples_dict[str(scalar)][min(idx, len(self.samples_dict[str(scalar)])-1)]
        return data_sample, scalar

    def __getitem__(self, item):
        item_content, interp_ratio = self.get_key_with_weights(item)

        sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts, loss_weight = self.data_loading(item_content, sample_t, interp_ratio)
        h, w = im0.shape[1:]
        if self.crop_size:
            if self.crop_size<h and self.crop_size<w:
                hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
                im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                           ws:ws + self.crop_size], events[
                                                                                                                    :,
                                                                                                                    hs:hs + self.crop_size,
                                                                                                                    ws:ws + self.crop_size]
                gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
            elif h < self.crop_size < w:
                hs, ws = (h // 32) * 32, random.randint(0, w - self.crop_size)
                hleft = (h - hs) // 2
                im0, im1, events = im0[:, hleft:hleft + hs, ws:ws + self.crop_size], im1[:, hleft:hleft + hs,ws:ws + self.crop_size], events[:,hleft:hleft + hs,ws:ws + self.crop_size]
                gts = [gt[:, hleft:hleft + hs, ws:ws + self.crop_size] for gt in gts]
            elif w < self.crop_size < h:
                hs, ws = random.randint(0, h - self.crop_size), (w // 32) * 32
                wleft = (w - ws) // 2
                im0, im1, events = im0[:, hs:hs + self.crop_size, wleft:wleft + ws], im1[:, hs:hs + self.crop_size,wleft:wleft + ws], events[:,hs:hs + self.crop_size,wleft:wleft + ws]
                gts = [gt[:, hs:hs + self.crop_size, wleft:wleft + ws] for gt in gts]
            else:
                hn, wn = (h // 32) * 32, (w // 32) * 32
                hleft = (h - hn) // 2
                wleft = (w - wn) // 2
                im0, im1, events = im0[:, hleft:hleft + hn, wleft:wleft + wn], im1[:, hleft:hleft + hn,
                                                                               wleft:wleft + wn], events[:,
                                                                                                  hleft:hleft + hn,
                                                                                                  wleft:wleft + wn]
                gts = [gt[:, hleft:hleft + hn, wleft:wleft + wn] for gt in gts]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': os.path.split(folder_name)[-1],
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio': interp_ratio,
            'loss_weight': loss_weight
        }
        return data_back

# 添加了图像亮度增强，根据事件数选择loss_weight，添加crop_range裁剪原始dataset，增大crop_size为512
class EnhancedMixLoader(Dataset):
    def __init__(self, params, training=True, verify_the_specified_folder=None):
        super().__init__()
        self.params = params
        self.dataset_path = params.args.dataset_path
        self.training_flag = training
        self.crop_size = params.training_config.crop_size if self.training_flag else None
        self.train_split_ratio = params.training_config.train_split_ratio
        self.test_dataset = params.training_config.test_dataset
        self.rgb_sampling_ratio = 1
        self.toim = ToPILImage()
        self.totensor = ToTensor()
        self.data_paths = {}
        if verify_the_specified_folder is not None and self.training_flag is False:
            selected_folders = [verify_the_specified_folder]
        else:
            # 划分数据集
            self.train_folders, self.test_folders = self.split_train_test()
            # 根据 training_flag 选择要加载的子集
            selected_folders = self.train_folders if self.training_flag else self.test_folders
        self.get_data_paths(selected_folders)
        self.samples_dict = {}
        self.interp_ratio_list = list(params.training_config.interp_ratio_list) if training else [params.validation_config.interp_ratio]
        self.interp_list_pob = list(params.training_config.interp_list_pob) if training else [1]
        # self.interp_list_pob = [1, 0, 0, 0] if training else [1]
        for irl in self.interp_ratio_list:
            self.samples_dict.update({
                str(irl):[]
            })
        self.samples_indexing()
        print(f'---- Interp List: {self.interp_ratio_list}, prob: {self.interp_list_pob}')
        self.norm_voxel = True
        self.echannel = params.model_config.define_model.echannel
        self.crop_range = params.training_config.crop_range
        self.transform = SyncColorJitter(brightness=0.2, contrast=0.2)
        self.event_num_threshold = params.training_config.event_num_threshold
        self.background_loss_weight = params.training_config.background_loss_weight

    def split_train_test(self):
        """划分训练和测试文件夹"""
        sub_folders = sorted(
            f for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f)) and re.match(r'^dataset_\d+$', f)
        )
        random.seed(self.params.args.seed)
        random.shuffle(sub_folders)
        if self.train_split_ratio is not None:
            split_index = int(len(sub_folders) * self.train_split_ratio)
            train_folders = sub_folders[:split_index]
            test_folders = sub_folders[split_index:]
        else:
            test_folders = self.test_dataset
            train_folders = [d for d in sub_folders if d not in test_folders]
        print(f"Total folders: {len(sub_folders)}, Train: {len(train_folders)}, Test: {len(test_folders)}")
        print("Train folders:", train_folders)
        print("Test folders:", test_folders)
        return train_folders, test_folders

    def get_data_paths(self, sub_folders):
        """读取指定文件夹下的所有 Frames 和 Events 文件位置"""
        for sub_folder in sub_folders:
            self.data_paths[sub_folder] = [[], []]
            self.data_paths[sub_folder][0] = [f for f in os.listdir(os.path.join(self.dataset_path,sub_folder,'Frames')) if
                              f.lower().endswith(".png")]
            self.data_paths[sub_folder][0].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][0] = [os.path.join(self.dataset_path,sub_folder,'Frames',f) for f in self.data_paths[sub_folder][0]]
            self.data_paths[sub_folder][1] = [f for f in os.listdir(os.path.join(self.dataset_path, sub_folder, 'Events')) if
                              f.lower().endswith(".npz")]
            self.data_paths[sub_folder][1].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][1] = [os.path.join(self.dataset_path, sub_folder, 'Events', f) for f in self.data_paths[sub_folder][1]]

    def samples_indexing(self):
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]

            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            for irl in self.interp_ratio_list:
                for i_ind in range(0, len(indexes) - irl, 1 if self.training_flag else irl):
                    rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + irl + 1]]
                    evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + irl]]
                    rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                    self.samples_dict[str(irl)].append([k, rgb_name, rgb_sample, evs_sample])
        return

    def __len__(self):
        return len(self.samples_dict[str(self.interp_ratio_list[0])])

    def imreader(self, impath):
        im = self.totensor(Image.open(impath).convert('RGB'))
        return im

    def events_reader(self, events_path, h, w, interp_ratio):
        evs_data = [np.load(ep) for ep in events_path]
        num_events = 0
        for ev in evs_data:
            num_events += len(ev['x'])
        if num_events<=self.event_num_threshold:
            loss_weight = self.background_loss_weight
        else:
            loss_weight = 1
        evs_voxels = []
        for ed in evs_data:
            evs_voxels.append(
                sample_events_to_grid(self.echannel//interp_ratio, h, w, ed['x'], ed['y'], ed['t'],
                                      ed['p']))
        return torch.tensor(np.concatenate(evs_voxels, 0)), loss_weight

    def data_loading(self, paths, sample_t, interp_ratio):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events, loss_weight = self.events_reader(evs_sample, h, w, interp_ratio)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts, loss_weight

    def transform_and_crop(self, folder_name, im0, im1, gts, events):
        # 根据crop_range裁剪，减少背景
        if folder_name in self.crop_range.keys():
            im0 = im0[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
            im1 = im1[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
            events = events[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
            gts = [gt[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]] for gt in gts]
        # 数据增强，调整亮度
        imgs_aug = self.transform([im0,im1]+gts)
        im0, im1 = imgs_aug[:2]
        gts = imgs_aug[2:]
        return im0, im1, gts, events

    def get_key_with_weights(self, idx):
        scalar = np.random.choice(self.interp_ratio_list, p=self.interp_list_pob)
        data_sample = self.samples_dict[str(scalar)][min(idx, len(self.samples_dict[str(scalar)])-1)]
        return data_sample, scalar

    def __getitem__(self, item):
        item_content, interp_ratio = self.get_key_with_weights(item)

        sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts, loss_weight = self.data_loading(item_content, sample_t, interp_ratio)
        if self.training_flag:
            im0, im1, gts, events = self.transform_and_crop(folder_name, im0, im1, gts, events)
        h, w = im0.shape[1:]
        if self.crop_size:
            if self.crop_size<h and self.crop_size<w:
                hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
                im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                           ws:ws + self.crop_size], events[
                                                                                                                    :,
                                                                                                                    hs:hs + self.crop_size,
                                                                                                                    ws:ws + self.crop_size]
                gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
            elif h < self.crop_size < w:
                hs, ws = (h // 32) * 32, random.randint(0, w - self.crop_size)
                hleft = (h - hs) // 2
                im0, im1, events = im0[:, hleft:hleft + hs, ws:ws + self.crop_size], im1[:, hleft:hleft + hs,ws:ws + self.crop_size], events[:,hleft:hleft + hs,ws:ws + self.crop_size]
                gts = [gt[:, hleft:hleft + hs, ws:ws + self.crop_size] for gt in gts]
            elif w < self.crop_size < h:
                hs, ws = random.randint(0, h - self.crop_size), (w // 32) * 32
                wleft = (w - ws) // 2
                im0, im1, events = im0[:, hs:hs + self.crop_size, wleft:wleft + ws], im1[:, hs:hs + self.crop_size,wleft:wleft + ws], events[:,hs:hs + self.crop_size,wleft:wleft + ws]
                gts = [gt[:, hs:hs + self.crop_size, wleft:wleft + ws] for gt in gts]
            else:
                hn, wn = (h // 32) * 32, (w // 32) * 32
                hleft = (h - hn) // 2
                wleft = (w - wn) // 2
                im0, im1, events = im0[:, hleft:hleft + hn, wleft:wleft + wn], im1[:, hleft:hleft + hn,
                                                                               wleft:wleft + wn], events[:,
                                                                                                  hleft:hleft + hn,
                                                                                                  wleft:wleft + wn]
                gts = [gt[:, hleft:hleft + hn, wleft:wleft + wn] for gt in gts]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': os.path.split(folder_name)[-1],
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio': interp_ratio,
            'loss_weight': loss_weight
        }
        return data_back

# 在 EnhancedMixLoader 基础上添加事件Mask
class EnhancedMixLoaderWithMask(Dataset):
    def __init__(self, params, training=True, verify_the_specified_folder=None):
        super().__init__()
        self.params = params
        self.dataset_path = params.args.dataset_path
        self.training_flag = training
        self.crop_size = params.training_config.crop_size if self.training_flag else None
        self.train_split_ratio = params.training_config.train_split_ratio
        self.test_dataset = params.training_config.test_dataset
        self.rgb_sampling_ratio = 1
        self.toim = ToPILImage()
        self.totensor = ToTensor()
        self.data_paths = {}
        if verify_the_specified_folder is not None and self.training_flag is False:
            selected_folders = [verify_the_specified_folder]
        else:
            # 划分数据集
            self.train_folders, self.test_folders = self.split_train_test()
            # 根据 training_flag 选择要加载的子集
            selected_folders = self.train_folders if self.training_flag else self.test_folders
        self.get_data_paths(selected_folders)
        self.samples_dict = {}
        self.interp_ratio_list = list(params.training_config.interp_ratio_list) if training else [params.validation_config.interp_ratio]
        self.interp_list_pob = list(params.training_config.interp_list_pob) if training else [1]
        # self.interp_list_pob = [1, 0, 0, 0] if training else [1]
        for irl in self.interp_ratio_list:
            self.samples_dict.update({
                str(irl):[]
            })
        self.samples_indexing()
        print(f'---- Interp List: {self.interp_ratio_list}, prob: {self.interp_list_pob}')
        self.norm_voxel = True
        self.echannel = params.model_config.define_model.echannel
        self.crop_range = params.training_config.crop_range
        self.transform = SyncColorJitter(brightness=0.2, contrast=0.2)
        self.event_num_threshold = params.training_config.event_num_threshold
        self.background_loss_weight = params.training_config.background_loss_weight

    def split_train_test(self):
        """划分训练和测试文件夹"""
        sub_folders = sorted(
            f for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f)) and re.match(r'^dataset_\d+$', f)
        )
        random.seed(self.params.args.seed)
        random.shuffle(sub_folders)
        if self.train_split_ratio is not None:
            split_index = int(len(sub_folders) * self.train_split_ratio)
            train_folders = sub_folders[:split_index]
            test_folders = sub_folders[split_index:]
        else:
            test_folders = self.test_dataset
            train_folders = [d for d in sub_folders if d not in test_folders]
        print(f"Total folders: {len(sub_folders)}, Train: {len(train_folders)}, Test: {len(test_folders)}")
        print("Train folders:", train_folders)
        print("Test folders:", test_folders)
        return train_folders, test_folders

    def get_data_paths(self, sub_folders):
        """读取指定文件夹下的所有 Frames 和 Events 文件位置"""
        for sub_folder in sub_folders:
            self.data_paths[sub_folder] = [[], []]
            self.data_paths[sub_folder][0] = [f for f in os.listdir(os.path.join(self.dataset_path,sub_folder,'Frames')) if
                              f.lower().endswith(".png")]
            self.data_paths[sub_folder][0].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][0] = [os.path.join(self.dataset_path,sub_folder,'Frames',f) for f in self.data_paths[sub_folder][0]]
            self.data_paths[sub_folder][1] = [f for f in os.listdir(os.path.join(self.dataset_path, sub_folder, 'Events')) if
                              f.lower().endswith(".npz")]
            self.data_paths[sub_folder][1].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][1] = [os.path.join(self.dataset_path, sub_folder, 'Events', f) for f in self.data_paths[sub_folder][1]]

    def samples_indexing(self):
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]

            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            for irl in self.interp_ratio_list:
                for i_ind in range(0, len(indexes) - irl, 1 if self.training_flag else irl):
                    rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + irl + 1]]
                    evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + irl]]
                    rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                    self.samples_dict[str(irl)].append([k, rgb_name, rgb_sample, evs_sample])
        return

    def __len__(self):
        return len(self.samples_dict[str(self.interp_ratio_list[0])])

    def imreader(self, impath):
        im = self.totensor(Image.open(impath).convert('RGB'))
        return im

    def events_reader(self, events_path, h, w, interp_ratio):
        evs_data = [np.load(ep) for ep in events_path]
        num_events = 0
        for ev in evs_data:
            num_events += len(ev['x'])
        if num_events<=self.event_num_threshold:
            loss_weight = self.background_loss_weight
        else:
            loss_weight = 1

        evs_voxels = []
        mask = np.zeros((len(evs_data)-1, h, w), dtype=np.float32)

        for i,ed in enumerate(evs_data):
            evs_voxels.append(
                sample_events_to_grid(self.echannel//interp_ratio, h, w, ed['x'], ed['y'], ed['t'],
                                      ed['p']))
            mask_x, mask_y = ed['x'], ed['y']
            if i!=len(evs_data)-1:
                mask[i, mask_y, mask_x] = 1.0
        return torch.tensor(np.concatenate(evs_voxels, 0)), loss_weight, torch.tensor(mask)

    def data_loading(self, paths, sample_t, interp_ratio):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events, loss_weight, mask = self.events_reader(evs_sample, h, w, interp_ratio)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts, loss_weight, mask

    def transform_and_crop(self, folder_name, im0, im1, gts, events, mask):
        # 根据crop_range裁剪，减少背景
        if folder_name in self.crop_range.keys():
            im0 = im0[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
            im1 = im1[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
            events = events[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
            gts = [gt[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]] for gt in gts]
            mask = mask[:, self.crop_range[folder_name][1]:self.crop_range[folder_name][3],
                  self.crop_range[folder_name][0]:self.crop_range[folder_name][2]]
        # 数据增强，调整亮度
        imgs_aug = self.transform([im0,im1]+gts)
        im0, im1 = imgs_aug[:2]
        gts = imgs_aug[2:]
        return im0, im1, gts, events, mask

    def get_key_with_weights(self, idx):
        scalar = np.random.choice(self.interp_ratio_list, p=self.interp_list_pob)
        data_sample = self.samples_dict[str(scalar)][min(idx, len(self.samples_dict[str(scalar)])-1)]
        return data_sample, scalar

    def __getitem__(self, item):
        item_content, interp_ratio = self.get_key_with_weights(item)

        sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts, loss_weight, mask = self.data_loading(item_content, sample_t, interp_ratio)
        if self.training_flag:
            im0, im1, gts, events, mask = self.transform_and_crop(folder_name, im0, im1, gts, events, mask)
        h, w = im0.shape[1:]
        if self.crop_size:
            if self.crop_size<h and self.crop_size<w:
                hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
                im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                           ws:ws + self.crop_size], events[
                                                                                                                    :,
                                                                                                                    hs:hs + self.crop_size,
                                                                                                                    ws:ws + self.crop_size]
                gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
                mask = mask[:, hs:hs + self.crop_size, ws:ws + self.crop_size]
            elif h < self.crop_size < w:
                hs, ws = (h // 32) * 32, random.randint(0, w - self.crop_size)
                hleft = (h - hs) // 2
                im0, im1, events = im0[:, hleft:hleft + hs, ws:ws + self.crop_size], im1[:, hleft:hleft + hs,ws:ws + self.crop_size], events[:,hleft:hleft + hs,ws:ws + self.crop_size]
                gts = [gt[:, hleft:hleft + hs, ws:ws + self.crop_size] for gt in gts]
                mask = mask[:, hleft:hleft + hs, ws:ws + self.crop_size]
            elif w < self.crop_size < h:
                hs, ws = random.randint(0, h - self.crop_size), (w // 32) * 32
                wleft = (w - ws) // 2
                im0, im1, events = im0[:, hs:hs + self.crop_size, wleft:wleft + ws], im1[:, hs:hs + self.crop_size,wleft:wleft + ws], events[:,hs:hs + self.crop_size,wleft:wleft + ws]
                gts = [gt[:, hs:hs + self.crop_size, wleft:wleft + ws] for gt in gts]
                mask = mask[:, hs:hs + self.crop_size, wleft:wleft + ws]
            else:
                hn, wn = (h // 32) * 32, (w // 32) * 32
                hleft = (h - hn) // 2
                wleft = (w - wn) // 2
                im0, im1, events = im0[:, hleft:hleft + hn, wleft:wleft + wn], im1[:, hleft:hleft + hn,
                                                                               wleft:wleft + wn], events[:,
                                                                                                  hleft:hleft + hn,
                                                                                                  wleft:wleft + wn]
                gts = [gt[:, hleft:hleft + hn, wleft:wleft + wn] for gt in gts]
                mask = mask[:, hleft:hleft + hn, wleft:wleft + wn]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': os.path.split(folder_name)[-1],
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio': interp_ratio,
            'loss_weight': loss_weight,
            'event_mask': mask
        }
        return data_back



# only for validation and testing no gt images
class MixLoader_without_gt(Dataset):
    def __init__(self, dataset_path,training=True, verify_the_specified_folder=None,
                 echannel = 128, train_interp_ratio=(2,4,8,16), interp_list_pob=(0.15,0.5,0.25,0.1),
                 test_interp_ratio=4, crop_size=None, train_split_ratio=0.8):
        super().__init__()
        self.dataset_path = dataset_path
        self.training_flag = training
        self.crop_size = crop_size if self.training_flag else None
        self.train_split_ratio = train_split_ratio
        self.rgb_sampling_ratio = 1
        self.toim = ToPILImage()
        self.totensor = ToTensor()
        self.data_paths = {}
        if verify_the_specified_folder is not None and self.training_flag is False:
            selected_folders = [verify_the_specified_folder]
        else:
            # 划分数据集
            self.train_folders, self.test_folders = self.split_train_test()
            # 根据 training_flag 选择要加载的子集
            selected_folders = self.train_folders if self.training_flag else self.test_folders
        self.get_data_paths(selected_folders)
        self.samples_dict = {}
        self.interp_ratio_list = list(train_interp_ratio) if training else [test_interp_ratio]
        self.interp_list_pob = list(interp_list_pob) if training else [1]
        # self.interp_list_pob = [1, 0, 0, 0] if training else [1]
        for irl in self.interp_ratio_list:
            self.samples_dict.update({
                str(irl):[]
            })
        self.samples_indexing()
        print(f'---- Interp List: {self.interp_ratio_list}, prob: {self.interp_list_pob}')
        self.norm_voxel = True
        self.echannel = echannel
        self.transform = ColorJitter(brightness=0.2, contrast=0.2)

    def split_train_test(self):
        """划分训练和测试文件夹"""
        sub_folders = sorted(
            f for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f)) and re.match(r'^dataset_\d+$', f)
        )
        random.shuffle(sub_folders)
        split_index = int(len(sub_folders) * self.train_split_ratio)
        train_folders = sub_folders[:split_index]
        test_folders = sub_folders[split_index:]
        print(f"Total folders: {len(sub_folders)}, Train: {len(train_folders)}, Test: {len(test_folders)}")
        print("Train folders:", train_folders)
        print("Test folders:", test_folders)
        return train_folders, test_folders

    def get_data_paths(self, sub_folders):
        """读取指定文件夹下的所有 Frames 和 Events 文件位置"""
        for sub_folder in sub_folders:
            self.data_paths[sub_folder] = [[], []]
            self.data_paths[sub_folder][0] = [f for f in os.listdir(os.path.join(self.dataset_path,sub_folder,'Frames')) if
                              f.lower().endswith(".png")]
            self.data_paths[sub_folder][0].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][0] = [os.path.join(self.dataset_path,sub_folder,'Frames',f) for f in self.data_paths[sub_folder][0]]
            self.data_paths[sub_folder][1] = [f for f in os.listdir(os.path.join(self.dataset_path, sub_folder, 'Events')) if
                              f.lower().endswith(".npz")]
            self.data_paths[sub_folder][1].sort(key=lambda x: int(re.search(r"(\d+)", x).group()))
            self.data_paths[sub_folder][1] = [os.path.join(self.dataset_path, sub_folder, 'Events', f) for f in self.data_paths[sub_folder][1]]

    def samples_indexing(self):
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]

            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            for irl in self.interp_ratio_list:
                for i_ind in range(0, len(indexes) - irl, 1 if self.training_flag else irl):
                    rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + irl + 1]]
                    evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + irl]]
                    rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                    self.samples_dict[str(irl)].append([k, rgb_name, rgb_sample, evs_sample])
        return

    def __len__(self):
        return len(self.samples_dict[str(self.interp_ratio_list[0])])

    def imreader(self, impath):
        im = Image.open(impath).convert('RGB')
        im = self.totensor(im)
        return im

    def events_reader(self, events_path, h, w, interp_ratio):
        evs_data = [np.load(ep) for ep in events_path]
        evs_voxels = []
        for ed in evs_data:
            evs_voxels.append(
                sample_events_to_grid(self.echannel//interp_ratio, h, w, ed['x'], ed['y'], ed['t'],
                                      ed['p']))
        return torch.tensor(np.concatenate(evs_voxels, 0))

    def data_loading(self, paths, sample_t, interp_ratio):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events = self.events_reader(evs_sample, h, w, interp_ratio)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def get_key_with_weights(self, idx):
        scalar = np.random.choice(self.interp_ratio_list, p=self.interp_list_pob)
        data_sample = self.samples_dict[str(scalar)][min(idx, len(self.samples_dict[str(scalar)])-1)]
        return data_sample, scalar

    def __getitem__(self, item):
        item_content, interp_ratio = self.get_key_with_weights(item)

        sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t, interp_ratio)
        h, w = im0.shape[1:]
        if self.crop_size:
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size], events[
                                                                                                                :,
                                                                                                                hs:hs + self.crop_size,
                                                                                                                ws:ws + self.crop_size]
        else:
            hn, wn = (h // 32) * 32, (w // 32) * 32
            hleft = (h - hn) // 2
            wleft = (w - wn) // 2
            im0, im1, events = im0[:, hleft:hleft + hn, wleft:wleft + wn], im1[:, hleft:hleft + hn,
                                                                           wleft:wleft + wn], events[:,
                                                                                              hleft:hleft + hn,
                                                                                              wleft:wleft + wn]
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': os.path.split(folder_name)[-1],
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio': interp_ratio
        }
        return data_back
