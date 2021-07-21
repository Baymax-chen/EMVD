from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import cv2
import warnings
warnings.filterwarnings('ignore')
from dataset import *
import config      as cfg
import time
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import structure
from torch.nn import functional as F


def test_big_size_raw(input_data, block_size, denoiser, a, b):
	stack_image = input_data
	hgt = np.shape(stack_image)[1]
	wid = np.shape(stack_image)[2]

	border = 32

	expand_raw = np.zeros(shape=[1, int(hgt * 2.5), int(wid * 2.5), 8], dtype=np.float)


	expand_raw[:,0:hgt, 0:wid,:] = stack_image
	expand_raw[:,0: hgt, border * 2: wid + border * 2,:] = stack_image
	expand_raw[:,border * 2:hgt + border * 2, 0:0 + wid + 0,:] = stack_image
	expand_raw[:,border * 2:hgt + border * 2, 0 + border * 2:0 + wid + border * 2,:] = stack_image

	expand_raw[:,0 + border:0 + border + hgt, 0 + border:0 + border + wid,:] = stack_image

	expand_res = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)
	expand_fusion = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)
	expand_denoise = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)
	expand_gamma = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),1], dtype=np.float)
	expand_omega = np.zeros([1, int(hgt * 2.5), int(wid * 2.5), 1], dtype=np.float)

	'''process'''
	for i in range(0 + border, hgt + border, int(block_size)):
		index = '%.2f' % (float(i) / float(hgt + border) * 100)
		print('run model : ', index, '%')
		for j in range(0 + border, wid + border, int(block_size)):
			block = expand_raw[:,i - border:i + block_size + border, j - border:j + block_size + border,:]   # t frame input
			block = preprocess(block).float()
			input = block

			with torch.no_grad():
				gamma, fusion_out, denoise_out, omega, refine_out= denoiser(input, a, b)
				fusion_out = tensor2numpy(fusion_out)
				refine_out = tensor2numpy(refine_out)
				denoise_out = tensor2numpy(denoise_out)
				gamma = tensor2numpy(F.upsample(gamma, scale_factor=2))
				omega = tensor2numpy(F.upsample(omega, scale_factor=2))
				expand_res[:,i:i + block_size, j:j + block_size,:] = refine_out[:,border:-border, border:-border,:]
				expand_fusion[:,i:i + block_size, j:j + block_size,:] = fusion_out[:,border:-border, border:-border,:]
				expand_denoise[:,i:i + block_size, j:j + block_size,:] = denoise_out[:,border:-border, border:-border,:]
				expand_gamma[:,i:i + block_size, j:j + block_size,:] = gamma[:,border:-border, border:-border,0:1]
				expand_omega[:, i:i + block_size, j:j + block_size, :] = omega[:, border:-border, border:-border, 0:1]

	refine_result = expand_res[:,border:hgt + border, border:wid + border,:]
	fusion_result = expand_fusion[:,border:hgt + border, border:wid + border,:]
	denoise_result = expand_denoise[:,border:hgt + border, border:wid + border,:]
	gamma_result = expand_gamma[:,border:hgt + border, border:wid + border,:]
	omega_result = expand_omega[:, border:hgt + border, border:wid + border, :]
	print('------------- Run Model Successfully -------------')

	return refine_result, fusion_result, denoise_result, gamma_result, omega_result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ngpu = cfg.ngpu
cudnn.benchmark = True

'''network'''
checkpoint = torch.load(cfg.best_model_save_root)
model = structure.MainDenoise()
model = model.to(device)
model.load_state_dict(checkpoint['model'])


# multi gpu test
if torch.cuda.is_available() and ngpu > 1:
	model = nn.DataParallel(model, device_ids=list(range(ngpu)))

model.eval()
output_dir = cfg.output_root

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

iso_list = [1600, 3200, 6400, 12800, 25600]

isp = torch.load('isp/ISP_CNN.pth')
iso_average_raw_psnr = 0
iso_average_raw_ssim = 0

# for iso_ind, iso in enumerate(iso_list):
for iso_ind in range(0,len(iso_list)):
	iso = iso_list[iso_ind]
	print('processing iso={}'.format(iso))

	if not os.path.isdir(output_dir + 'ISO{}'.format(iso)):
		os.makedirs(output_dir + 'ISO{}'.format(iso))

	f = open('denoise_model_test_psnr_and_ssim_on_iso{}.txt'.format(iso), 'w')

	context = 'ISO{}'.format(iso) + '\n'
	f.write(context)

	scene_avg_raw_psnr = 0
	scene_avg_raw_ssim = 0
	frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
	a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
	b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

	for scene_id in range(7,11+1):
		context = 'scene{}'.format(scene_id) + '\n'
		f.write(context)

		frame_avg_raw_psnr = 0
		frame_avg_raw_ssim = 0
		block_size = 512
		ft0_fusion_data = np.zeros([1, 540, 960, 4 * 7])
		gt_fusion_data = np.zeros([1, 540, 960, 4 * 7])
		for time_ind in range(0,7):
			raw_name = os.path.join(cfg.data_root[1],'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy0.tiff'.format(scene_id, scene_id, iso, frame_list[time_ind]))
			raw = cv2.imread(raw_name, -1)
			input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)

			gt_raw = cv2.imread(os.path.join(cfg.data_root[1],
												  'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
													  scene_id,scene_id, iso, frame_list[time_ind])), -1).astype(np.float32)
			fgt = np.expand_dims(pack_gbrg_raw(gt_raw), axis=0)

			if time_ind == 0:
				ft0_fusion = input_full  # 1 * 512 * 512 * 4
			else:
				ft0_fusion = ft0_fusion_data[:, :, :,  (time_ind-1) * 4: (time_ind) * 4]  # 1 * 512 * 512 * 4

			input_data = np.concatenate([ft0_fusion, input_full], axis=3)
			coeff_a = a_list[iso_ind] / (2 ** 12 - 1 - 240)
			coeff_b = b_list[iso_ind] / (2 ** 12 - 1 - 240) ** 2
			refine_out, fusion_out, denoise_out, gamma_out, omega_out = test_big_size_raw(input_data, block_size, model, coeff_a, coeff_b)

			ft0_fusion_data[:, :, :,  time_ind * 4: (time_ind+1) * 4] = fusion_out

			test_result = depack_gbrg_raw(refine_out)
			test_fusion = depack_gbrg_raw(fusion_out)
			test_denoise = depack_gbrg_raw(denoise_out)

			test_gt = (gt_raw - 240) / (2 ** 12 - 1 - 240)

			test_raw_psnr = compare_psnr(test_gt, (
						np.uint16(test_result * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
													 2 ** 12 - 1 - 240), data_range=1.0)
			test_raw_ssim = compute_ssim_for_packed_raw(test_gt, (
						np.uint16(test_result * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
																	2 ** 12 - 1 - 240))
			test_raw_psnr_input = compare_psnr(test_gt, (raw - 240) / (2 ** 12 - 1 - 240), data_range=1.0)
			print('scene {} frame{} test raw psnr : {}, test raw input psnr : {}, test raw ssim : {} '.format(scene_id, time_ind, test_raw_psnr, test_raw_psnr_input, test_raw_ssim))
			context = 'raw psnr/ssim: {}/{}, input_psnr:{}'.format(test_raw_psnr, test_raw_ssim, test_raw_psnr_input) + '\n'
			f.write(context)
			frame_avg_raw_psnr += test_raw_psnr
			frame_avg_raw_ssim += test_raw_ssim

			output = test_result * (2 ** 12 - 1 - 240) + 240
			fusion = test_fusion * (2 ** 12 - 1 - 240) + 240
			denoise = test_denoise * (2 ** 12 - 1 - 240) + 240


			if cfg.vis_data:
				noisy_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(raw), axis=0))
				noisy_srgb_frame = tensor2numpy(isp(noisy_raw_frame))[0]
				gt_raw_frame = np.expand_dims(pack_gbrg_raw(test_gt * (2 ** 12 - 1 - 240) + 240), axis=0)
				gt_srgb_frame = tensor2numpy(isp(preprocess(gt_raw_frame)))[0]
				cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_noisy_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(noisy_srgb_frame * 255))
				cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_gt_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(gt_srgb_frame * 255))

			denoised_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(output), axis=0))
			denoised_srgb_frame = tensor2numpy(isp(denoised_raw_frame))[0]
			cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_denoised_sRGB.png'.format(iso, scene_id, time_ind),
						np.uint8(denoised_srgb_frame * 255))

			if cfg.vis_data:
				fusion_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(fusion), axis=0))
				fusion_srgb_frame = tensor2numpy(isp(fusion_raw_frame))[0]
				cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_fusion_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(fusion_srgb_frame * 255))

				denoise_midres_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(denoise), axis=0))
				denoised_mid_res_srgb_frame = tensor2numpy(isp(denoise_midres_raw_frame))[0]
				cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_denoised_midres_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(denoised_mid_res_srgb_frame * 255))

				cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_gamma.png'.format(iso, scene_id, time_ind), np.uint8(gamma_out[0] * 255))
				cv2.imwrite(output_dir + 'ISO{}/scene{}_frame{}_omega.png'.format(iso, scene_id, time_ind),
							np.uint8(omega_out[0] * 255))
				print('gamma.max:', gamma_out.max())
				print('gamma.min:', gamma_out.min())

		frame_avg_raw_psnr = frame_avg_raw_psnr / 7
		frame_avg_raw_ssim = frame_avg_raw_ssim / 7
		context = 'frame average raw psnr:{},frame average raw ssim:{}'.format(frame_avg_raw_psnr,
																			   frame_avg_raw_ssim) + '\n'
		f.write(context)

		scene_avg_raw_psnr += frame_avg_raw_psnr
		scene_avg_raw_ssim += frame_avg_raw_ssim

	scene_avg_raw_psnr = scene_avg_raw_psnr / 5
	scene_avg_raw_ssim = scene_avg_raw_ssim / 5
	context = 'scene average raw psnr:{},scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
																				 scene_avg_raw_ssim) + '\n'
	print(context)
	f.write(context)
	iso_average_raw_psnr += scene_avg_raw_psnr
	iso_average_raw_ssim += scene_avg_raw_ssim

iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)

context = 'iso average raw psnr:{},iso frame average raw ssim:{}'.format(iso_average_raw_psnr, iso_average_raw_ssim) + '\n'
f.write(context)
print(context)



