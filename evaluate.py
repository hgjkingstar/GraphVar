import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


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



batch_size = 32
train_loader = DataLoader(combined_dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(combined_dataset_test, batch_size=batch_size, shuffle=False)


feature_input_dim = combined_dataset_train.feature_dataset[0][0].shape[1]
print(feature_input_dim)

num_classes = len(combined_dataset_train.label_encoder.classes_)
print(num_classes)
model = MultiModalModelWithTransformer(num_classes=num_classes,num_sparse_features=feature_input_dim,transformer_input_dim=feature_input_dim)
model.load_state_dict(torch.load('./model_info/final_model.pt'))
model.eval()
# 在测试数据集上运行模型并获取预测结果
true_labels = []
predicted_labels = []

with torch.no_grad():
    for image_inputs, feature_inputs, labels in tqdm(test_loader):
        feature_inputs = feature_inputs.permute(1, 0, 2)
        outputs = model(image_inputs, feature_inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())


# 计算各类指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# 将结果存储在DataFrame中
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]
})

print(metrics_df)

