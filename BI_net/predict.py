from matplotlib import pyplot as plt
from numpy import interp
from sklearn import metrics
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from utils import calculate, _init_fn, ConfusionMatrix
from Datasets import CustomDataset
from MS_biformerV6 import MS_biformer


# 定义一个函数用于绘制ROC曲线
def plot_roc_curve(mean_fpr, tprs, aucs, mtype_list, tpr_list, fpr_list, Task_name):
    plt.figure()
    fold_all = len(mtype_list)

    save_path_plot = "C:\\Users\\whd\\PycharmProjects\\AD&HC\\comvfomer2_3\\result"

    for i, mtype in enumerate(mtype_list):
        plt.plot(fpr_list[i], tpr_list[i], lw=1.6, alpha=0.7,
                 label='%s (AUC = %0.4f)' % (mtype, aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=0.8, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve')
    plt.rcParams.update({'font.size': 7})
    plt.legend(loc="lower right")
    plt.legend(fontsize='9')
    plt.savefig("{}/{}ROC.png".format(save_path_plot, Task_name), dpi=300)
    plt.show()


def predict(weights_fold_th, label_list, test_data_retriever, save_path, model):
    print(weights_fold_th)
    data_loader = torch_data.DataLoader(
        test_data_retriever,
        batch_size=8,  # SIZE=4, 8
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=_init_fn
    )

    model.to(device)
    confusion = ConfusionMatrix(num_classes=2, labels=label_list)
    checkpoint = torch.load(weights_fold_th, map_location='cuda:0')
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    y_pred = []
    targets_all = []
    y_all = []

    for e, batch in enumerate(data_loader, 1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = torch.sigmoid(model(batch[0].to(device))[0]).cpu().numpy().squeeze()
            targets = batch[1]
            targets_all.extend(targets.numpy().tolist())
            targets.to(device)
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())

            y_all.extend(batch[1].tolist())

    fpr_micro, tpr_micro, th = metrics.roc_curve(y_all, y_pred)
    max_th = -1
    max_yd = -1
    for i in range(len(th)):
        yd = tpr_micro[i] - fpr_micro[i]
        if yd > max_yd:
            max_yd = yd
            max_th = th[i]
    print(f'max_th: {max_th}')
    max_th_save = os.path.join(save_path, 'max_th')
    with open(max_th_save, 'w') as f:
        f.write(str(max_th))

    y_preint = [1 if x > max_th else 0 for x in y_pred]
    confusion.update(np.array(y_preint), np.array(targets_all, dtype=np.int32))
    confusion.plot(save_path)
    confusion.summary()

    rst_val = pd.DataFrame([calculate(y_pred, y_all, max_th)])
    preddf = pd.DataFrame({"label": y_all, "y_pred": y_pred})
    return preddf, rst_val, y_all, y_pred  # 返回每一个病人的预测分数的表格和预测指标AUC，ACC，Sep，sen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_root_path = 'C:\\User\\whd\\PycharmProjects\\AD&HC\\comvfomer2_3'

Task_name = 'MCI'  # ADHC or MCI
roc_pictrue_name = "MVFEAblation"
all_name_roc = roc_pictrue_name + '_' + Task_name
weights_path = f"C:\\Users\\whd\\PycharmProjects\\AD&HC\\comvfomer2_3\\ablation_weights_{Task_name}"
# 加载保存的5折模型
modelfiles = []
for model_name in os.listdir(weights_path):     # model_name:f1,f2,...
    if model_name.endswith('.pth'):
        model_path_final = os.path.join(weights_path, model_name)
        modelfiles.append(model_path_final)

test_path_root = ''
labels = []
if Task_name == 'ADHC':
    test_path_root = 'E:\\datasets\\using_datasets\\112ad&hc_mmsebuquan\\data\\togather_image_to_sub\\merge_class_fold' \
                     f'\\fold_result'
    labels = ['HC', 'AD']  # 0 AND 1
else:
    test_path_root = 'E:\\datasets\\using_datasets\\112all_mmsebuquan_mci\\data\\togather_image_to_sub\\merge_class_fold' \
                     f'\\fold_result'
    labels = ['sMCI', 'pMCI']  # 0 AND 1

matric_save_path = os.path.join(model_root_path, "result")
os.makedirs(matric_save_path, exist_ok=True)

# 1，用5个模型对测试集数据进行预测，并取平均值
rst_test_all = []
scores = []
df_test = {}
fold_nums = len(modelfiles)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
mtype_list = []
fpr_list = []
tpr_list = []

# 实例化模型
model = MS_biformer()

for fold_th, weights_fold_th in enumerate(modelfiles):
    name = weights_fold_th.split("\\")[-1]
    mtype_list.append(name)
    test_path = os.path.join(test_path_root, 'test')
    test_data_retriever = CustomDataset(test_path, split='val')

    save_path = os.path.join(matric_save_path, f'{Task_name}_result', f"{name}")
    os.makedirs(save_path, exist_ok=True)

    preddf, rst_test, y_all, y_pred = predict(weights_fold_th, labels, test_data_retriever, save_path, model, fold_th)

    fpr, tpr, thresholds = roc_curve(y_all, y_pred)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = roc_auc_score(y_all, y_pred)
    aucs.append(roc_auc)
    rst_test_all.append(rst_test)
    df_test[name] = preddf["y_pred"]

# 调用绘制函数进行绘图
plot_roc_curve(np.linspace(0, 1, 100), tprs, aucs, mtype_list, tpr_list, fpr_list, all_name_roc)

rst_test_all = pd.concat(rst_test_all)
rst_test_all = pd.DataFrame(rst_test_all)
df_test['label'] = preddf["label"]
df_test = pd.DataFrame(df_test)
rst_test_all.loc['mean'] = rst_test_all.mean(axis=0)
rst_test_all.to_csv(os.path.join(matric_save_path, 'train_test_pf.csv'))
print('测试集{}折模型预测，取平均指标：{}'.format(fold_nums, rst_test_all))

# 2，对5折预测的分数取平均值，并计算指标
df_test = pd.DataFrame(df_test)
df_test["Average"] = 0
# for mtype in mri_types:
for i in range(0, fold_nums):
    df_test["Average"] += df_test.iloc[:, i]
df_test["Average"] /= fold_nums
df_test.to_csv(os.path.join(matric_save_path, 'test_score5.csv'))
auc = roc_auc_score(df_test["label"], df_test["Average"])
print(f"test ensemble AUC: {auc:.4f}")  # 整体平均值评估

fpr_micro, tpr_micro, th = metrics.roc_curve(df_test["label"], df_test["Average"])
max_th = -1
max_yd = -1
for i in range(len(th)):
    yd = tpr_micro[i] - fpr_micro[i]
    if yd > max_yd:
        max_yd = yd
        max_th = th[i]
print(max_th)

rst_test = pd.DataFrame([calculate(df_test["Average"], df_test["label"], max_th)])
rst_test.to_csv(os.path.join(matric_save_path, 'test_ensembel_res.csv'))
print('{}折分数取平均之后的测试集指标：{}'.format(fold_nums, rst_test))
print('{}折预测的分数，以及分数平均值表格：{}'.format(fold_nums, df_test))
