import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageFolder


class CombinedDataset(Dataset):
    def __init__(self, image_dir, feature_dir, transform=None, max_rows=150, max_cols=36):
        self.image_dataset = ImageFolder(image_dir, transform=transform)
        self.feature_dataset = self.load_features(feature_dir, max_rows, max_cols)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([label for _, label in self.feature_dataset])

    def load_features(self, data_dir, max_rows, max_cols):
        samples = []
        for i, disease_folder in enumerate(os.listdir(data_dir)):
            disease_path = os.path.join(data_dir, disease_folder)
            if os.path.isdir(disease_path):
                for sample_file in os.listdir(disease_path):
                    if sample_file.endswith('.csv'):
                        sample_path = os.path.join(disease_path, sample_file)
                        features = pd.read_csv(sample_path).values.astype(np.float32)
                        num_rows, num_cols = features.shape
                        if num_rows < max_rows:
                            padding_rows = max_rows - num_rows
                            padding = np.zeros((padding_rows, num_cols), dtype=np.float32)
                            features = np.concatenate((features, padding), axis=0)
                        elif num_rows > max_rows:
                            features = features[:max_rows, :]
                        if num_cols < max_cols:
                            padding_cols = max_cols - num_cols
                            padding = np.zeros((max_rows, padding_cols), dtype=np.float32)
                            features = np.concatenate((features, padding), axis=1)
                        elif num_cols > max_cols:
                            features = features[:, :max_cols]
                        samples.append((torch.tensor(features), i))
        return samples

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        features, label = self.feature_dataset[idx]
        return image, features, label
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=4, num_encoder_layers=3, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

    def forward(self, src):
        output = self.transformer(src)
        return output


class MultiModalModelWithTransformer(nn.Module):
    def __init__(self, num_classes, num_sparse_features, transformer_input_dim):
        super(MultiModalModelWithTransformer, self).__init__()
        self.image_branch = models.resnet18(pretrained=True)
        num_ftrs = self.image_branch.fc.in_features
        self.image_branch.fc = nn.Linear(num_ftrs, 128)

        self.transformer_branch = TransformerModel(transformer_input_dim)

        self.fc_final = nn.Linear(128 + 36, num_classes)

    def forward(self, image_input, transformer_input):
        image_output = self.image_branch(image_input)
        transformer_output = self.transformer_branch(transformer_input)
        transformer_output = transformer_output.mean(dim=0)
        transformer_output = transformer_output.view(transformer_output.size(0), -1)

        combined_output = torch.cat((image_output, transformer_output), dim=1)

        return self.fc_final(combined_output)
# 数据路径
image_data_train_dir = './imagedata/train'
image_data_test_dir = './imagedata/test'
feature_data_train_dir = './featuredata/train'
feature_data_test_dir = './featuredata/test'
combined_dataset_train = CombinedDataset(image_data_train_dir, feature_data_train_dir, transform=transforms.ToTensor())
combined_dataset_test = CombinedDataset(image_data_test_dir, feature_data_test_dir, transform=transforms.ToTensor())

# Grad-CAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子以获取特征图和梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, image_input, feature_input, target_class):
        self.model.eval()
        feature_input = feature_input.permute(1, 0, 2)
        # 获取模型的综合输出
        output = self.model(image_input, feature_input)

        # 计算目标类的损失
        loss = output[:, target_class].sum()

        # 反向传播以获取梯度
        self.model.zero_grad()
        loss.backward()

        # 获取梯度和特征图
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]

        # 计算权重并生成Grad-CAM
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 归一化并调整大小
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image_input.shape[2], image_input.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam
feature_input_dim = combined_dataset_train.feature_dataset[0][0].shape[1]
print(feature_input_dim)

num_classes = len(combined_dataset_train.label_encoder.classes_)
print(num_classes)
model = MultiModalModelWithTransformer(num_classes=num_classes,num_sparse_features=feature_input_dim,transformer_input_dim=feature_input_dim)
model.load_state_dict(torch.load('./model_info/final_model.pt'))
model.eval()
# 选择要可视化的层
target_layer = model.image_branch.layer4[-1]

# 创建Grad-CAM对象
grad_cam = GradCAM(model, target_layer)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(6.6, 3))
for idx, i in enumerate(range(0, 5)):

    image_input, feature_input, label = combined_dataset_train[i]

    target_class = label# 设置目标类，您可以更改为特定的类别
    cam = grad_cam(image_input.unsqueeze(0), feature_input.unsqueeze(0), target_class)
    # 应用阈值过滤
    threshold = 0.5
    # cam[cam < threshold] = 0

    # 将热力图叠加到原始图像上进行显示
    original_image = Image.fromarray(np.uint8(image_input.permute(1, 2, 0).numpy() * 255))  # 将张量转换为 PIL 图像格式
    original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)
      # 转换为RGB格式
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_output = heatmap + np.float32(original_image) / 255
    cam_output = cam_output / np.max(cam_output)
    # 2. 在指定的子图(axes[idx])上显示图像
    ax = axes[idx]
    ax.imshow(cam_output)
    ax.axis('off')  # 隐藏坐标轴，让图像更美观
plt.tight_layout()
plt.savefig("热力图.jpg",dpi=330)

