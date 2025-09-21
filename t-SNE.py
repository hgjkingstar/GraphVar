import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class CombinedDataset(Dataset):
    def __init__(self, image_dir, feature_dir, transform=None, max_rows=150, max_cols=36):
        self.image_dataset = ImageFolder(image_dir, transform=None)
        self.feature_dir = feature_dir
        self.transform = transform
        self.classes = self.image_dataset.classes
        self.max_rows = max_rows
        self.max_cols = max_cols

        print(f"Matching image and feature files in '{os.path.basename(image_dir)}'...")
        self.samples = []
        for img_path, label in self.image_dataset.samples:
            img_filename = os.path.basename(img_path)
            base_name_parts = img_filename.split('.maf_')
            if len(base_name_parts) > 0:
                base_name = base_name_parts[0]
                feature_filename = base_name + '.csv'
                feature_path = os.path.join(self.feature_dir, self.classes[label], feature_filename)

                if os.path.exists(feature_path):
                    self.samples.append(((img_path, feature_path), label))

        if not self.samples:
            print(f"Warning: No successfully matched image-feature pairs were found in the '{os.path.basename(image_dir)}' directory.")
        else:
            print(f"Matching complete! Found {len(self.samples)} valid sample pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (img_path, feature_path), label = self.samples[idx]

        image = self.image_dataset.loader(img_path)
        if self.transform:
            image = self.transform(image)

        features = pd.read_csv(feature_path).values.astype(np.float32)

        # Pad or truncate rows
        num_rows, num_cols = features.shape
        if num_rows < self.max_rows:
            padding_rows = self.max_rows - num_rows
            padding = np.zeros((padding_rows, num_cols), dtype=np.float32)
            features = np.concatenate((features, padding), axis=0)
        elif num_rows > self.max_rows:
            features = features[:self.max_rows, :]
        
        current_rows = features.shape[0]

        # Pad or truncate columns
        if num_cols < self.max_cols:
            padding_cols = self.max_cols - num_cols
            padding = np.zeros((current_rows, padding_cols), dtype=np.float32)
            features = np.concatenate((features, padding), axis=1)
        elif num_cols > self.max_cols:
            features = features[:, :self.max_cols]

        features_tensor = torch.tensor(features)

        return image, features_tensor, label


# --- Model Definition (consistent with training) ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=4, num_encoder_layers=3, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                       batch_first=True),
            num_layers=num_encoder_layers
        )

    def forward(self, src):
        return self.transformer(src)


class MultiModalModelWithTransformer(nn.Module):
    def __init__(self, num_classes, transformer_input_dim):
        super(MultiModalModelWithTransformer, self).__init__()
        self.image_branch = models.resnet18(pretrained=False)
        num_ftrs = self.image_branch.fc.in_features
        self.image_branch.fc = nn.Linear(num_ftrs, 128)
        self.transformer_branch = TransformerModel(transformer_input_dim)
        self.fc_final = nn.Linear(128 + transformer_input_dim, num_classes)

    def forward(self, image_input, transformer_input):
        image_output = self.image_branch(image_input)
        transformer_output = self.transformer_branch(transformer_input)
        transformer_output = transformer_output.mean(dim=1)
        combined_output = torch.cat((image_output, transformer_output), dim=1)
        final_output = self.fc_final(combined_output)
        return final_output


# --- Configure Paths and Parameters ---
MODEL_PATH = './model_info/final_model.pt'
IMAGE_DATA_DIR = './stratified_imagedata'
FEATURE_DATA_DIR = './stratified_featuredata'
BATCH_SIZE = 64
NUM_CLASSES = 33
TRANSFORMER_INPUT_DIM = 36

# --- Load Model and Set Hook ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultiModalModelWithTransformer(num_classes=NUM_CLASSES, transformer_input_dim=TRANSFORMER_INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

features_to_visualize = []


def hook_fn(module, input, output):
    features_to_visualize.append(input[0].cpu().detach().numpy())


model.fc_final.register_forward_hook(hook_fn)
print("Model loaded and forward hook registered on the final classification layer.")

# --- Define Data Loading and Feature Extraction Functions ---
data_transform = transforms.Compose([
    transforms.ToTensor()
])


def extract_features_multimodal(loader, model, device):
    global features_to_visualize
    features_to_visualize = []
    all_labels = []
    with torch.no_grad():
        for image_inputs, feature_inputs, labels in tqdm(loader, desc="Extracting Features"):
            image_inputs = image_inputs.to(device)
            feature_inputs = feature_inputs.to(device)
            _ = model(image_inputs, feature_inputs)
            all_labels.append(labels.numpy())

    # Check if the lists are empty before returning
    if not features_to_visualize or not all_labels:
        return None, None

    return np.concatenate(features_to_visualize, axis=0), np.concatenate(all_labels, axis=0)


# --- Load Data and Extract Features ---
train_dataset = CombinedDataset(os.path.join(IMAGE_DATA_DIR, 'train'), os.path.join(FEATURE_DATA_DIR, 'train'),
                                data_transform)
test_dataset = CombinedDataset(os.path.join(IMAGE_DATA_DIR, 'test'), os.path.join(FEATURE_DATA_DIR, 'test'),
                               data_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_features, train_labels = extract_features_multimodal(train_loader, model, device)
test_features, test_labels = extract_features_multimodal(test_loader, model, device)

# Check if features were extracted successfully
if train_features is None or test_features is None:
    print("\nError: Failed to extract features, possibly because no files were matched. Please check the warning messages above.")
    exit()

all_features = np.concatenate([train_features, test_features], axis=0)
all_labels = np.concatenate([train_labels, test_labels], axis=0)
set_labels = ['Train'] * len(train_features) + ['Test'] * len(test_features)
print("Feature extraction complete.")

# --- Run t-SNE Dimensionality Reduction ---
print("Running t-SNE dimensionality reduction, this may take a few minutes...")
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(all_features)
print("t-SNE dimensionality reduction complete.")

# --- Visualization ---
print("Generating visualization plot...")
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = tsne_results[:, 0]
df_tsne['tsne-2d-two'] = tsne_results[:, 1]
class_names = train_dataset.classes
df_tsne['Class'] = [class_names[i] for i in all_labels]
df_tsne['Set'] = set_labels

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Class", style="Set",
    palette=sns.color_palette("hsv", n_colors=NUM_CLASSES),
    data=df_tsne, legend="full", alpha=0.7
)
plt.title('t-SNE visualization of feature space', fontsize=16)
plt.xlabel('t-SNE dimension 1', fontsize=14)
plt.ylabel('t-SNE dimension 2', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig('./tsne_visualization_multimodal.pdf', dpi=330)
plt.show()