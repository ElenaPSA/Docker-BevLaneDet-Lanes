import os
import sys
sys.path.append('/data/gvincent/bev_lane_det/')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utilities.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, image_gt_segment=None, image_gt_instance=None, image_gt_category=None, train=True):
        pred_2d, emb_2d, category_2d = self.model(inputs)
        if train:
            ## 2d
            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)
            loss_category_2d = self.ce_loss(category_2d, image_gt_category)
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d + 1. * loss_category_2d
            # loss_total_2d = loss_total_2d.unsqueeze(0)
            return pred_2d, loss_total_2d
        else:
            return pred_2d


def train_epoch(model, dataset, optimizer, configs, epoch, writer):
    # Last iter as mean loss of whole epoch
    model.train()
    losses_avg = {}
    num_steps = len(dataset)
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data, image_gt_segment, image_gt_instance, image_gt_category) in enumerate(dataset):
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()
        image_gt_segment = image_gt_segment.cuda()
        image_gt_instance = image_gt_instance.cuda()
        image_gt_category = image_gt_category.cuda()
        prediction, loss_total_2d = model(input_data, image_gt_segment, image_gt_instance, image_gt_category)
        loss_back_2d = loss_total_2d.mean()
        ''' caclute loss '''

        optimizer.zero_grad()
        loss_back_2d.backward()
        optimizer.step()
        if idx % 50 == 0:
            loss_iter = {"2d loss":loss_back_2d.item()}
            print(idx, loss_iter, '*' * 10)
        if idx % 300 == 0:
            target = image_gt_segment.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            loss_iter = {"2d_loss":loss_back_2d.item(), "F1_BEV_seg": f1_bev_seg}
            print(idx, loss_iter)
            for k,v in loss_iter.items():
                writer.add_scalar(k,
                                  v,
                                  epoch*num_steps+idx)
    target = image_gt_segment.detach().cpu().numpy().ravel()
    pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
    f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
    loss_iter = {"epoch/2d_loss":loss_back_2d.item(), "epoch/F1_BEV_seg": f1_bev_seg}
    for k,v in loss_iter.items():
        writer.add_scalar(k,
                          v,
                          epoch)


def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is '+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)
    # os.makedirs(configs.log_path)

    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            print('resuming training...')
            resume_training(checkpoint_path, model.module, optimizer, scheduler, configs.start_epoch)
        else:
            print('loading checkpoint')
            load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    writer = SummaryWriter(configs.log_path)

    for epoch in range(configs.start_epoch, configs.epochs):
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, configs, epoch, writer)
        writer.add_scalar('epoch/lr',
                          scheduler.get_last_lr()[-1],
                          epoch)
        scheduler.step()
        save_model_dp(model, optimizer, configs.model_save_path, 'ep%03d.pth' % epoch)
        save_model_dp(model, None, configs.model_save_path, 'latest.pth')
    writer.close()


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('/data/gvincent/bev_lane_det/tools/openlane_config.py', gpu_id=[0,1,2,3])#, checkpoint_path='/data/gvincent/bev_lane_det/checkpoints/openlane/ep008.pth')