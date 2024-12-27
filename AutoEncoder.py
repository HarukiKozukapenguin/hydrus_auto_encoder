import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (1, 100, 100) -> (16, 50, 50)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (16, 50, 50) -> (32, 25, 25)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 25, 25) -> (64, 13, 13)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 13, 13) -> (128, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*7*7, 4096),  # 圧縮次元数
        )
        
        # デコーダ
        self.decoder = nn.Sequential(
            nn.Linear(4096, 128*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 7, 7) -> (64, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 14, 14) -> (32, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 28, 28) -> (16, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 56, 56) -> (1, 100, 100)
            nn.Sigmoid()  # 入力画像と同じ範囲 [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # 'L' mode for grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # 正規化 [0, 1]
        img = np.expand_dims(img, axis=0)  # チャネルを追加 (1, H, W)
        
        if self.transform:
            img = self.transform(img)
        
        return torch.tensor(img)

if __name__ == "__main__":
    # データセットとデータローダの作成
    image_dir = 'make_fig/fig_data'  # 画像ディレクトリ
    dataset = CustomDataset(image_dir)
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # モデル、損失関数、最適化アルゴリズムの定義
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 訓練ループ
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # 順伝播
            optimizer.zero_grad()
            outputs = model(data)
            
            # 損失の計算
            loss = criterion(outputs, data)
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 訓練後のモデルの保存（任意）
    torch.save(model.state_dict(), "hydrus_autoencoder_model.pth")
