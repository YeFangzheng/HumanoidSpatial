import torch
import numpy as np
import cv2 as cv
import pickle
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.optim import build_optim_wrapper
from mmengine.runner import autocast, load_state_dict
import time
import tqdm

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

config = 'configs/xhumanoid/fusionocc_concat_768x960.py'
cfg = Config.fromfile(config)

cfg.model._scope_ = 'mmdet3d'
cfg.train_dataloader.dataset._scope_ = 'mmdet3d'
cfg.train_dataloader.sampler._scope_ = 'mmdet3d'
cfg.val_dataloader.dataset._scope_ = 'mmdet3d'
cfg.val_dataloader.sampler._scope_ = 'mmdet3d'
# cfg.train_dataloader.sampler.shuffle = False
cfg.train_dataloader.batch_size = 1
# cfg.train_dataloader.sampler.samples_per_gpu = 1
cfg.train_dataloader.num_workers = 1
cfg.optim_wrapper.type = 'OptimWrapper'

# cfg.val_dataloader.dataset.ann_file = 'train_frames.json'
# cfg.val_dataloader.dataset.filter = ['68288d8a1efd00b21c143cb4']
# cfg.val_dataloader.dataset.filter = ['6834aa2288046d2265d8976f']


model = Runner.build_model(Runner, cfg.model)

# model.init_weights()
# print(sum(p.numel() for p in model.parameters()) / 1e6)

# ckpts = torch.load('work_dirs/fusionocc_concat_768x960/epoch_20.pth')
# ckpts = torch.load('exp/fusionocc_household_768x960_seq1/epoch_20.pth')
# load_state_dict(model, ckpts['state_dict'])
model.cuda()

optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

train = False
if train:
    data_loader = Runner.build_dataloader(cfg.train_dataloader)
else:
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model.eval()

# print(sum(p.numel() for p in model.parameters()))
outputs = []
# total = 0
for i, data in tqdm.tqdm(enumerate(data_loader)):
    if train:
        losses = model.train_step(data, optim_wrapper)
        # losses.backward()
    else:
        t0 = time.time()
        torch.cuda.synchronize()
        with autocast('cuda', enabled=False):
            with torch.no_grad():
                results = model.val_step(data)
        # t1 = time.time()
        # torch.cuda.synchronize()
        # if i > 0:
        #     total += t1 - t0
        #     print(t1 - t0, total / i, int(i / total))

        points = data['inputs']['points'][0][:, :3].numpy()
        imgs = data['data_samples'][0].ori_img.numpy()[:,:,:,::-1]
        pred_occ = results[0]['pred_occupancy'].cpu().numpy()
        gt_occ = results[0]['gt_occupancy'].cpu().numpy()

        ego2global = data['data_samples'][0].ego_pose.numpy()
        timestamp = data['data_samples'][0].timestamp
        cam2ego = data['data_samples'][0].cam2ego.numpy()
        cam2img = data['data_samples'][0].intrinsic.numpy()
        distortion = data['data_samples'][0].distortion.numpy()
        # print(i, data['data_samples'][0].scene_name, data['data_samples'][0].prev_exists, \
        #         data['data_samples'][1].scene_name, data['data_samples'][1].prev_exists)
        #         data['data_samples'][2].scene_name, data['data_samples'][2].frame_idx, \
        #         data['data_samples'][3].scene_name, data['data_samples'][3].frame_idx)
        # imgs = [cv.undistort(img, cam2img[i], distortion[i]) for i, img in enumerate(imgs)]
        # imgs = [cv.resize(img, (960, 768)) for img in imgs]
        output = dict(points=points, imgs=imgs, pred_occ=pred_occ, gt_occ=gt_occ, ego2global=ego2global, timestamp=timestamp, cam2ego=cam2ego, cam2img=cam2img, distortion=distortion)

        if i == 5:
            outputs.append(output)
            break


with open('outputs.pkl', 'wb') as f:
    pickle.dump(outputs, f)