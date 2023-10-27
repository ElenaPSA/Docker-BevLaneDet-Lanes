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

from torchvision.ops.focal_loss import  sigmoid_focal_loss as sigmoid_focal_loss

class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model,use_focal_loss=False):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.use_focal_loss=use_focal_loss
        print("focal loss : ",self.use_focal_loss)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, image_gt_category=None, image_gt_segment=None,
                image_gt_instance=None, train=True):
        
        pred_2d, emb_2d, category_2d = self.model(inputs)
       
        if train:
            
            if self.use_focal_loss:
                loss_seg_2d = sigmoid_focal_loss(pred_2d, image_gt_segment,alpha=0.9) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            else:
                loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
           # 
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)
            

            loss_category_2d = self.ce_loss(category_2d, image_gt_category)
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d + 1. * loss_category_2d
            
            
            return pred_2d, loss_total_2d,loss_seg_2d,loss_emb_2d,loss_category_2d
        else:
            return pred_2d

def eval(model, dataset):
    model.eval()
    
def train_epoch(model, dataset, optimizer, configs, epoch, writer):
    # Last iter as mean loss of whole epoch
    model.train()
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data,image_category,image_gt_segment, image_gt_instance) in enumerate(
            dataset):
        
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()
        image_category=image_category.cuda()
        image_gt_segment = image_gt_segment.cuda()
        image_gt_instance = image_gt_instance.cuda()
        pred_2d, loss_total_2d,loss_seg_2d,loss_emb_2d,loss_category_2d  = model(input_data,
                                                                                image_category,
                                                                                image_gt_segment,
                                                                                image_gt_instance)
        loss_back_2d = loss_total_2d.mean()
        loss_seg_2d=loss_seg_2d.mean()
        loss_emb_2d=loss_emb_2d.mean()
        loss_category_2d=loss_category_2d.mean()
        ''' caclute loss '''

        optimizer.zero_grad()
        loss_back_2d.backward()
        optimizer.step()
        if idx % 50 == 0:
            loss_iter = {"total loss":loss_back_2d.item(), "Seg Loss": loss_seg_2d.item(), 'Emb loss': loss_emb_2d.item(), 'category loss': loss_category_2d.item()}
            print(idx, loss_iter, '*' * 10)
            
        if idx % 300 == 0:
            target = image_gt_segment.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(pred_2d).detach().cpu().numpy().ravel()
            f1_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            loss_iter = {"total loss":loss_back_2d.item(), "Seg Loss": loss_seg_2d.item(), 'Emb loss': loss_emb_2d.item(), 'category loss': loss_category_2d.item(),
                            "F1_seg": f1_seg}
            print(idx, loss_iter)
            for k,v in loss_iter.items():
                writer.add_scalar(k,
                                  v,
                                  idx)

def validate(model, dataset, configs, epoch):

    model.eval()
    f1_seg_list = []
    print("Validation ")
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (input_data,image_category,image_gt_segment, image_gt_instance,_) in enumerate(dataset):

        input_data = input_data.cuda()
        
        image_gt_segment = image_gt_segment.cuda()
        image_category=image_category.cuda()
        image_gt_instance = image_gt_instance.cuda()
        prediction = model(input_data,
                            image_category,
                            image_gt_segment,
                            image_gt_instance,
                            False)
        target = image_gt_segment.detach().cpu().numpy().ravel()
        pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
        f1_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
        f1_seg_list.append(f1_seg)
        if idx % 50 == 0:
            print(idx, "Average f1  : ",sum(f1_seg_list)/len(f1_seg_list))

    return sum(f1_seg_list)/len(f1_seg_list)

def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is '+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' models and optimizer '''
    model = configs.model()
    usefocalloss=configs.usefocalloss
    model = Combine_Model_and_Loss(model,usefocalloss)
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
    if configs.balance:
        dataset, sampler = getattr(configs, "train_dataset", None)
        configs.loader_args['shuffle'] = False
        train_loader = DataLoader(dataset(), **configs.loader_args, pin_memory=True, sampler=sampler)
    else:
        dataset = getattr(configs, "train_dataset", None)
        train_loader = DataLoader(dataset(), **configs.loader_args, pin_memory=True)

    ''' get validation '''
    if configs.with_validation:
        val_dataset = configs.val_dataset()
        val_loader = DataLoader(dataset=val_dataset,
                            batch_size=8,
                            num_workers=8,
                            shuffle=False)
        
    writer = SummaryWriter(configs.log_path)
    best_val=0.0
    for epoch in range(configs.start_epoch, configs.epochs):
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, configs, epoch, writer)
        writer.add_scalar('epoch/lr',
                          scheduler.get_last_lr()[-1],
                          epoch)
        scheduler.step()
        if configs.with_validation:
            f1_val=validate(model, val_loader, configs,epoch)
            writer.add_scalar("F1_seg_val",f1_val,epoch)
            if f1_val>=best_val:
                best_val=f1_val
                save_model_dp(model, None, configs.model_save_path, 'best.pth')
                
        save_model_dp(model, optimizer, configs.model_save_path, 'ep%03d.pth' % epoch)
        save_model_dp(model, None, configs.model_save_path, 'latest.pth')
    writer.close()


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('./bev_lane_det/tools/stellantis_gt_2d_config.py', gpu_id=[0,1,2,3])
