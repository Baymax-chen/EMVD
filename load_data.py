import os
import cv2
import numpy as np
import config as cfg
from dataset import *
from torch.utils.data import Dataset
import time
iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

def load_cvrd_data(shift, noisy_level, scene_ind, frame_ind, xx, yy):

	frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]

	gt_name = os.path.join(cfg.data_root[1],
						   'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
							   scene_ind, scene_ind, iso_list[noisy_level],
							   frame_list[frame_ind + shift]))
	gt_raw = cv2.imread(gt_name, -1)
	gt_raw_full = gt_raw
	gt_raw_patch = gt_raw_full[yy:yy + cfg.image_height * 2,
				   xx:xx + cfg.image_width * 2]  # 256 * 256
	gt_raw_pack = np.expand_dims(pack_gbrg_raw(gt_raw_patch), axis=0)  # 1* 128 * 128 * 4

	noisy_frame_index_for_current = np.random.randint(0, 10)
	input_name = os.path.join(cfg.data_root[1],
							  'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(
								  scene_ind, scene_ind, iso_list[noisy_level],
								  frame_list[frame_ind + shift], noisy_frame_index_for_current))
	noisy_raw = cv2.imread(input_name, -1)
	noisy_raw_full = noisy_raw
	noisy_patch = noisy_raw_full[yy:yy + cfg.image_height * 2, xx:xx + cfg.image_width * 2]
	input_pack = np.expand_dims(pack_gbrg_raw(noisy_patch), axis=0)
	return input_pack, gt_raw_pack


def load_eval_data(noisy_level, scene_ind):
	input_batch_list = []
	gt_raw_batch_list = []

	input_pack_list = []
	gt_raw_pack_list = []

	xx = 200
	yy = 200

	for shift in range(0, cfg.frame_num):
		# load gt raw
		frame_ind = 0
		input_pack, gt_raw_pack = load_cvrd_data(shift, noisy_level, scene_ind, frame_ind, xx, yy)
		input_pack_list.append(input_pack)
		gt_raw_pack_list.append(gt_raw_pack)

	input_pack_frames = np.concatenate(input_pack_list, axis=3)
	gt_raw_pack_frames = np.concatenate(gt_raw_pack_list, axis=3)

	input_batch_list.append(input_pack_frames)
	gt_raw_batch_list.append(gt_raw_pack_frames)

	input_batch = np.concatenate(input_batch_list, axis=0)
	gt_raw_batch = np.concatenate(gt_raw_batch_list, axis=0)

	in_data = torch.from_numpy(input_batch.copy()).permute(0, 3, 1, 2).cuda()  # 1 * (4*25) * 128 * 128
	gt_raw_data = torch.from_numpy(gt_raw_batch.copy()).permute(0, 3, 1, 2).cuda()  # 1 * (4*25) * 128 * 128
	return in_data, gt_raw_data

def generate_file_list(scene_list):
	iso_list = [1600, 3200, 6400, 12800, 25600]
	file_num = 0
	data_name = []
	for scene_ind in scene_list:
		for iso in iso_list:
			for frame_ind in range(1,8):
				gt_name = os.path.join('ISO{}/scene{}_frame{}_gt_sRGB.png'.format(
								 iso, scene_ind, frame_ind-1))
				data_name.append(gt_name)
				file_num += 1

	random_index = np.random.permutation(file_num)
	data_random_list = []
	for i,idx in enumerate(random_index):
		data_random_list.append(data_name[idx])
	return data_random_list

def read_img(img_name, xx, yy):
	raw = cv2.imread(img_name, -1)
	raw_full = raw
	raw_patch = raw_full[yy:yy + cfg.image_height * 2,
				   xx:xx + cfg.image_width * 2]  # 256 * 256
	raw_pack_data = pack_gbrg_raw(raw_patch)
	return raw_pack_data

def decode_data(data_name):
	frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
	H = 1080
	W = 1920
	xx = np.random.randint(0, (W - cfg.image_width * 2 + 1) / 2) * 2
	yy = np.random.randint(0, (H - cfg.image_height * 2 + 1) / 2) * 2

	scene_ind = data_name.split('/')[1].split('_')[0]
	frame_ind = int(data_name.split('/')[1].split('_')[1][5:])
	iso_ind = data_name.split('/')[0]

	noisy_level_ind = iso_list.index(int(iso_ind[3:]))
	noisy_level = [a_list[noisy_level_ind], b_list[noisy_level_ind]]

	gt_name_list = []
	noisy_name_list = []
	xx_list = []
	yy_list = []
	for shift in range(0, cfg.frame_num):
		gt_name = os.path.join(cfg.data_root[1],'indoor_raw_gt/indoor_raw_gt_{}/{}/{}/frame{}_clean_and_slightly_denoised.tiff'.format(
							   scene_ind,scene_ind,iso_ind,frame_list[frame_ind + shift]))

		noisy_frame_index_for_current = np.random.randint(0, 10)
		noisy_name = os.path.join(cfg.data_root[1],
								  'indoor_raw_noisy/indoor_raw_noisy_{}/{}/{}/frame{}_noisy{}.tiff'.format(
									  scene_ind,scene_ind, iso_ind, frame_list[frame_ind + shift], noisy_frame_index_for_current))

		gt_name_list.append(gt_name)
		noisy_name_list.append(noisy_name)

		xx_list.append(xx)
		yy_list.append(yy)

	gt_raw_data_list  = list(map(read_img, gt_name_list, xx_list, yy_list))
	noisy_data_list = list(map(read_img, noisy_name_list, xx_list, yy_list))
	gt_raw_batch = np.concatenate(gt_raw_data_list, axis=2)
	noisy_raw_batch = np.concatenate(noisy_data_list, axis=2)

	return noisy_raw_batch, gt_raw_batch, noisy_level


class loadImgs(Dataset):
	def __init__(self, filelist):
		self.filelist = filelist

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, item):
		self.data_name = self.filelist[item]
		image, label, noisy_level = decode_data(self.data_name)
		self.image = image
		self.label = label
		self.noisy_level = noisy_level
		return self.image, self.label, self.noisy_level
