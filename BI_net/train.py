from torch.optim import lr_scheduler
from sklearn import metrics
import os
import time
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
from Datasets import CustomDataset
from sklearn.metrics import roc_auc_score
from utils import mkdir, calculate, _init_fn, set_seed, make_train_pic, \
    save_model_super_state, Focal_Loss, WarmupMultiStepLR
from model1 import FusionNet

set_seed(12)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


def warmup_lr(current_step, warmup_steps, base_lr):
    return base_lr * (current_step / warmup_steps)


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion,
            RESUME,
            start_model_save_epoch,
            init_lr,
            train_nums,
            val_nums,
            scheduler,
            train_type='byacc',
    ):
        self.train_type = train_type
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = 0  # np.inf
        self.n_patience = 0
        self.lastmodel = None
        self.RESUME_path = RESUME
        self.init_lr = init_lr
        self.start_model_save_epoch = start_model_save_epoch
        self.train_nums = train_nums
        self.val_nums = val_nums
        self.scheduler = scheduler
        self.acc_best = 0.0
        self.loss_smallest = 0.0

    def fit(self, epochs, train_loader, valid_loader, modility, save_path, floder_path, patience, fold, BS):
        val_acc_list = []
        train_acc_list = []
        val_losses_list = []
        train_losses_list = []
        start_epoch = 0

        # 从上次的训练停止epoch开始训练
        if self.RESUME_path:
            checkpoint = torch.load(os.path.join(save_path, self.RESUME_path))  # 加载断点
            self.model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
            start_epoch = checkpoint['n_epoch']  # 设置开始的epoch
            self.acc_best = checkpoint['current_val_acc']  # 中断前的最佳acc记录
            print("best_acc: {}".format(self.acc_best))
            self.best_valid_score = checkpoint['best_valid_score']  # 中断前的最佳auc记录
            print("best_auc: {}".format(self.best_valid_score))
            print("current_loss: {}".format(checkpoint['current_val_loss']))
            val_acc_list = checkpoint['val_acc_list']

        for n_epoch in range(start_epoch + 1, epochs + 1):
            self.info_message("EPOCH: {}/{}", n_epoch, epochs)
            train_loss, train_auc, train_time, rst_train, acc_train = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time, rst_val, acc_val = self.valid_epoch(valid_loader)

            self.scheduler.step()

            train_acc_list.append(acc_train)  # for plot
            val_acc_list.append(acc_val)
            train_losses_list.append(train_loss)
            val_losses_list.append(valid_loss)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, auc: {:.4f},time: {:.2f} s ",
                n_epoch, train_loss, train_auc, train_time
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            if self.train_type == 'byacc':
                bool_for_different_traintype = acc_val >= self.acc_best and\
                                               n_epoch >= self.start_model_save_epoch
            elif self.train_type == 'byloss':
                bool_for_different_traintype = valid_loss <= self.loss_smallest and\
                                               n_epoch >= self.start_model_save_epoch

            if bool_for_different_traintype:
                # 调用保存参数函数
                self.save_model(n_epoch, modility, save_path, floder_path, current_val_loss=valid_loss, auc=valid_auc,
                                fold=fold, all_epochs=epochs, patience=patience, current_val_acc=acc_val, BS=BS,
                                val_acc_list=val_acc_list, val_losses_list=val_losses_list)
                # 打印信息
                self.info_message(
                    "AUC from {:.4f} to {:.4f}. ACC from {:.4f} to {:.4f}.Saved model to '{}'",
                    self.best_valid_score, valid_auc, self.acc_best, acc_val, self.lastmodel
                )
                self.best_valid_score = valid_auc
                self.acc_best = acc_val
                self.loss_smallest = valid_loss
                self.n_patience = 0         # 此时置位早停忍耐次数
                final_rst_train = rst_train
                final_rst_val = rst_val
            else:
                self.n_patience += 1    # 性能没改善，此时早停忍耐次数 + 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

        # 绘图代码
        make_train_pic(modility, fold, self.acc_best, train_acc_list, train_losses_list, val_acc_list,
                       folder_path=floder_path, save_path=save_path)

        all_rst = [final_rst_train, final_rst_val]
        rst = pd.concat(all_rst, axis=1)
        print(rst)
        return rst

    def train_epoch(self, train_loader):

        self.model.train()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(train_loader, 1):
            X = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            outputs = outputs.squeeze(1)

            # w = [1.788, 1.]  # 标签0和标签1的权重
            # weight = torch.zeros(targets.shape)  # 权重矩阵
            # weight.to(self.device)
            # for i in range(targets.shape[0]):
            #     weight[i] = w[int(targets[i])]

            loss = self.criterion(outputs, targets)
            loss.to(device)
            loss.backward()

            sum_loss += loss.detach().item()
            y_all.extend(batch[1].tolist())
            outputs_all.extend(outputs.tolist())

            self.optimizer.step()

            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss / step, end="\r")

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)
        fpr_micro, tpr_micro, th = metrics.roc_curve(y_all, outputs_all)
        max_th = -1
        max_yd = -1
        for i in range(len(th)):
            yd = tpr_micro[i] - fpr_micro[i]
            if yd > max_yd:
                max_yd = yd
                max_th = th[i]
        acc = calculate(outputs_all, y_all, max_th)['acc']
        rst_train = pd.DataFrame([calculate(outputs_all, y_all, max_th)])

        return sum_loss / len(train_loader), auc, int(time.time() - t), rst_train, acc

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(X)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch[1].tolist())
                outputs_all.extend(outputs.tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss / step, end="\r")

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)
        fpr_micro, tpr_micro, th = metrics.roc_curve(y_all, outputs_all)
        max_th = -1
        max_yd = -1
        for i in range(len(th)):
            yd = tpr_micro[i] - fpr_micro[i]
            if yd > max_yd:
                max_yd = yd
                max_th = th[i]
        acc = calculate(outputs_all, y_all, max_th)['acc']
        rst_val = pd.DataFrame([calculate(outputs_all, y_all, max_th)])

        return sum_loss / len(valid_loader), auc, int(time.time() - t), rst_val, acc

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)

    def save_model(self, n_epoch, modility, save_path, floder_path, current_val_loss, auc, fold, all_epochs, patience,
                   current_val_acc, BS, val_acc_list, val_losses_list):

        model_name = f"{floder_path}-fold{fold}.pth"
        model_name_txt = model_name[:-4] + ".txt"
        self.lastmodel = os.path.join(save_path, model_name)
        torch.save(
            {
                "device": device,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_milestones": self.scheduler.milestones,
                "lr_scheduler_gamma": self.scheduler.gamma,
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                "auc": auc,
                "stop_model_checkpoint": self.lastmodel,
                "all_epoch": all_epochs,
                "init_lr": self.init_lr,
                "patience": patience,
                "current_val_acc": current_val_acc,
                "current_val_loss": current_val_loss,
                "train_datasets": self.train_nums,
                "val_datasets": self.val_nums,
                "batch_size": BS,
                "val_acc_list": val_acc_list,
                "val_losses_list": val_losses_list,
                "space": '\n\n',

            },
            self.lastmodel,
        )
        # 保存参数txt
        save_model_super_state(save_path, self.lastmodel, model_name_txt)


def train_mri_type(mri_type, data_path, model_save_path, model_floder, fold_path_name_list, RESUME=None,
                   start_model_save_epoch=None):
    fold_root_path = os.path.join(model_save_path, model_floder)
    fold_path = [os.path.join(data_path, f) for f in fold_path_name_list]
    print(fold_path)
    for fold, fold_dir in enumerate(fold_path):
        print(f'---------Fold{fold + 1} training start!-----------:')
        rst_dfs = []
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'test')

        result_save_path = os.path.join(fold_root_path, f'fold{fold + 1}')
        mkdir(result_save_path)

        if not start_model_save_epoch:
            start_model_save_epoch = 3

        train_data_retriever = CustomDataset(train_dir, split='train')
        valid_data_retriever = CustomDataset(val_dir, split='val')
        train_dataset_nums = len(train_data_retriever)
        val_dataset_nums = len(valid_data_retriever)
        print(f"nums of train datasets: {train_dataset_nums}\nnums of val datasets: {val_dataset_nums}")

        epoch = 140
        patience = 15
        batch_size = 8
        model = FusionNet()
        model.to(device)
        lr = 0.00005
        # ------scheduler para----------
        milestones = [80]
        gamma = 0.5
        # ------------------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # weight_decay=1e-4
        criterion = torch_functional.binary_cross_entropy_with_logits
        # criterion = SmoothBCEWithLogitsLoss(smoothing=0.05)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)  # or [40, 65]

        train_loader = torch_data.DataLoader(
            train_data_retriever,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            worker_init_fn=_init_fn,
            drop_last=True
        )

        valid_loader = torch_data.DataLoader(
            valid_data_retriever,
            batch_size=batch_size,  # SIZE=4, 8
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            worker_init_fn=_init_fn,
            drop_last=True
        )

        trainer = Trainer(
            model,
            device,
            optimizer,
            criterion,
            RESUME,
            start_model_save_epoch,
            init_lr=lr,
            train_nums=train_dataset_nums,
            val_nums=val_dataset_nums,
            scheduler=scheduler

        )

        rst = trainer.fit(
            epoch,
            train_loader,
            valid_loader,
            f"{mri_type}",
            result_save_path,
            model_floder,
            patience,
            fold=fold,
            BS=batch_size
        )

        rst_dfs.append(rst)
        rst_dfs = pd.concat(rst_dfs)
        print(rst_dfs)
        rst_dfs = pd.DataFrame(rst_dfs)
        rst_dfs.to_csv(os.path.join(result_save_path, 'train_val_res_pf.csv'))  # 保存每一折的指标
        print('fold ' + str(fold + 1) + ' finished!')


if __name__ == "__main__":
    # 映射/home/wuhuidong/whd/3dLenet_AD_HC
    target_classification = 1
    methon_model_name = 'doubleNet'
    Exper_name = 'newstemv2_5fold_model1'
    fold_path_name_list = ['f1', 'f2', 'f3', 'f4', 'f5']
    target_name: str = ''
    if target_classification == 0:
        print("AD VS HC")
        target_name = "AD_VS_HC"
        datasets_root = f"/home/wuhuidong/datasets/adhc_data"
    elif target_classification == 1:
        print('sMCI VS pMCI')
        target_name = "sMCI_VS_pMCI"
        datasets_root = f"/home/wuhuidong/datasets/mci_data"
    elif target_classification == 2:
        print('sMCI VS HC')
        target_name = "sMCI_VS_HC"
        datasets_root = ""
    elif target_classification == 3:
        print('sMCI VS AD')
        target_name = "sMCI_VS_AD"
        datasets_root = ""
    elif target_classification == 4:
        print('pMCI VS HC')
        target_name = "pMCI_VS_HC"
        datasets_root = ""
    elif target_classification == 5:
        print('pMCI VS AD')
        target_name = "pMCI_VS_HC"
        datasets_root = ""
    elif target_classification == 6:
        print('local_test_HC VS AD')
        target_name = "local_test"
        datasets_root = "E:\\datasets\\using_datasets\\112ad&hc_mmsebuquan\\data\\togather_image_to_sub" \
                        "\\merge_class_fold\\fold_result"
    model_name = f'{target_name}_lr1e-4_milestones_40_gamma_0.5_have_aug'
    model_root_path = f'{methon_model_name}_cloud/{target_name}_{len(fold_path_name_list)}fold/{Exper_name}'
    # ---------------对上次训练异常中断的结果继续训练------------------
    resume_model = False   # it's True when need resume
    fold_stop = 0  # int,如果只是训练单折被异常终止，则此处为0，如果为K折，则此处为中断的那一折数字
    RESUME = ''
    if resume_model:
        RESUME = model_name + f"-fold{fold_stop}"+".pth"  # str
    else:
        RESUME = None
    # ------------------------------------------------------------------------------------------------
    train_mri_type(target_name, datasets_root, model_root_path, model_name, fold_path_name_list,
                   start_model_save_epoch=1,
                   RESUME=RESUME)
