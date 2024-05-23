from typing import List
import bisect
from bisect import bisect_right
from matplotlib import pyplot as plt
from numpy import interp
from prettytable import PrettyTable
from skimage import transform, exposure
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import torch
from sklearn.metrics import roc_curve, auc
from dataset_aug import *

# 加载npy数据和label
from sklearn.metrics import roc_auc_score, roc_curve


def load_npy_data(data_dir, split):
    datanp = []  # images
    truenp = []  # labels
    for file in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        #         data[0][0] = resize(data[0][0], (224,224,224))
        if split == 'train':
            # 各种方式各扩充一次，共为原数据集大小的四倍
            data_sug = transform.rotate(data[0][0], 60)  # 旋转60度，不改变大小
            data_sug2 = exposure.exposure.adjust_gamma(data[0][0], gamma=0.5)  # 变亮
            # data_sug3 = data_add.randomflip(data[0][0])
            # data_sug4 = data_add.noisy(0.005)
            datanp.append(data_sug)
            truenp.append(data[0][1])
            datanp.append(data_sug2)
            truenp.append(data[0][1])
            # datanp.append(data_sug3)
            # truenp.append(data[0][1])
            # datanp.append(data_sug4)
            # truenp.append(data[0][1])

        datanp.append(data[0][0])
        truenp.append(data[0][1])
    datanp = np.array(datanp)
    # numpy.array可使用 shape。list不能使用shape。可以使用np.array(list A)进行转换。
    # 不能随意加维度
    datanp = np.expand_dims(datanp, axis=4)  # 加维度,from(256,256,128)to(256,256,128,1),according the cnn tabel.png
    datanp = datanp.transpose(0, 4, 1, 2, 3)
    truenp = np.array(truenp)
    print(datanp.shape, truenp.shape)
    # print(np.min(datanp), np.max(datanp), np.mean(datanp), np.median(datanp))
    return datanp, truenp


# def load_npy_data(data_dir, split, k=3):
#     data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
#     if split == 'train':
#         data_name = 'train_data.dat'
#         label_name = 'train_label.dat'
#     else:
#         data_name = 'val_data.dat'
#         label_name = 'val_label.dat'
#     datanp = np.memmap(data_name, dtype='float32', mode='w+', shape=(k * len(data_files), 1, 128, 128, 128))
#     truenp = np.memmap(label_name, dtype='int64', mode='w+', shape=(k * len(data_files),))
#
#     for i, file in enumerate(data_files):
#         data = np.load(file, allow_pickle=True)
#         if split == 'train':
#             data_sug = transform.rotate(data[0][0], 60)
#             data_sug2 = exposure.exposure.adjust_gamma(data[0][0], gamma=0.5)
#             datanp[3 * i, 0, :, :, :] = data_sug
#             datanp[3 * i + 1, 0, :, :, :] = data_sug2
#             truenp[3 * i] = data[0][1]
#             truenp[3 * i + 1] = data[0][1]
#             datanp[3 * i + 2, 0, :, :, :] = data[0][0]
#             truenp[3 * i + 2] = data[0][1]
#         else:
#             datanp[i, 0, :, :, :] = data[0][0]
#             truenp[i] = data[0][1]
#
#     return datanp, truenp


# 定义随机种子
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(12) + worker_id)


# 计算分类的各项指标
def calculate(score, label, th):
    score = np.array(score)
    label = np.array(label)
    pred = np.zeros_like(label)
    pred[score >= th] = 1
    pred[score < th] = 0

    TP = len(pred[(pred > 0.5) & (label > 0.5)])
    FN = len(pred[(pred < 0.5) & (label > 0.5)])
    TN = len(pred[(pred < 0.5) & (label < 0.5)])
    FP = len(pred[(pred > 0.5) & (label < 0.5)])

    AUC = metrics.roc_auc_score(label, score)
    result = {'AUC': AUC, 'acc': (TP + TN) / (TP + TN + FP + FN), 'sen': TP / (TP + FN + 0.0001),
              'spe': TN / (TN + FP + 0.0001)}
    #     print('acc',(TP+TN),(TP+TN+FP+FN),'spe',(TN),(TN+FP),'sen',(TP),(TP+FN))
    return result


def make_roc_pic(score, label, fpr, tpr, title='ROC'):
    AUC = metrics.roc_auc_score(label, score)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % AUC)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def make_rocs(roc_auc, fpr, tpr):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold  (AUC = %0.2f)' % roc_auc)


def make_train_pic(modility, fold, acc_best, train_acc_list, train_losses_list, val_acc_list, folder_path, val_losses_list=None,
                   save_path=None):
    # 绘图代码
    plt.figure()  # 创建新的图形窗口
    plt.plot(np.arange(len(train_losses_list)), train_losses_list, label="train loss")
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
    plt.plot(np.arange(len(val_losses_list)), val_losses_list, label="valid loss")
    plt.plot(np.arange(len(val_acc_list)), val_acc_list, label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.title('Model accuracy&loss of fold{}, acc={:.4f}'.format(fold, acc_best))
    plt.title(f"{modility} Model accuracy&loss")
    plt.savefig("./{}/{}_fold{}_acc&loss.png".format(save_path, folder_path,  fold))
    # plt.show()
    plt.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_model_super_state(save_root, model_path, name):
    path = os.path.join(save_root, name)
    net = torch.load(model_path, map_location=torch.device('cpu'))
    for key, value in net.items():
        if key != 'model_state_dict' and key != 'optimizer_state_dict':
            with open(path, 'a') as f:
                str = f'{key}: {value}\n'
                f.write(str)

        if key == 'optimizer_state_dict':
            opti = value['param_groups']
            with open(path, 'a') as f:
                str = f'{opti}\n'
                f.write(str)


class Focal_Loss():
    """
    二分类Focal Loss
    """

def __init__(self, alpha=0.64, gamma=4):
    super(Focal_Loss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma


def forward(self, preds, labels):
    """
    preds:sigmoid的输出结果
    labels：标签
    """
    eps = 1e-7
    loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
    loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
    loss = loss_0 + loss_1
    return torch.mean(loss)

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self, save_path):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig("{}/Confusionmatrix .png".format(save_path))
        plt.show()

    def plot_roc_curve(self, y_prob, y_test):
        fpr, tpr, th = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
##################################################
## self.base_lrs 【0.001，.... 0.001】 len = 84
#################################################
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()



def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
