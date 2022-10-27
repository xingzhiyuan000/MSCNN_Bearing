import os
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from nets.WCCNN import *
import matplotlib.pyplot as plt
from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np
import json


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

    def plot(self):
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
        plt.show()


if __name__ == '__main__':

    model_path = ".\models\WCCNN_A_Dropout0.4_1000.pth"  # 预测模型路径
    #定义训练的设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #加载自制数据集

    # root = "./testset/0"  # 数据集所在根目录
    # root = "./testset/1"  # 数据集所在根目录
    # root = "./testset/2"  # 数据集所在根目录
    root = "./testset/3"  # 数据集所在根目录

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    #添加tensorboard
    #writer=SummaryWriter("logs",flush_secs=5)


    test_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                              transform="1")

    test_data_size=len(test_data_set)

    #加载数据集
    batch_size = 64
    test_dataloader = torch.utils.data.DataLoader(test_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=test_data_set.collate_fn)

    #加载网络模型
    model=torch.load(model_path)
    model=model.to(device) #将模型加载到cuda

    #读取 class_indict的json文件并获取类别便签
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=13, labels=labels) #设置类别数量

    #test_real_lable = [] #存储测试集的真实标签
    total_correct_num=0 #总体的正确率
    model.eval() #设置为测试模式
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            imgs, targets = data
            imgs = imgs.type(torch.cuda.FloatTensor)
            #test_real_lable.append(targets.numpy())
            imgs = imgs.to(device)  # 将图片加载到cuda上训练
            targets = targets.to(device)  # 加载到cuda上训练
            outputs = model(imgs)

            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), targets.to("cpu").numpy())

            #correct_num = (outputs.argmax(1) == targets).sum()  # 1:表示横向取最大值所在项
            #total_correct_num = total_correct_num + correct_num  # 计算预测正确的总数
    #print("测试集总体正确率为: {}".format(total_correct_num / test_data_size))
    confusion.plot()
    confusion.summary()

    #writer.flush()
    #writer.close()
