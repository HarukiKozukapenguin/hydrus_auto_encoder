import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# 自作のAutoEncoderクラスをインポート
from AutoEncoder import AutoEncoder  # もしくはクラス定義があるファイル名を指定

# 学習済みモデルのパス
MODEL_PATH = 'hydrus_autoencoder_model.pth'

# 画像を読み込み、前処理する変換（リサイズとテンソル化）
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 画像を112x112にリサイズ
    transforms.ToTensor(),          # Tensorに変換
])

# 推論用のデータセットクラス（画像を1枚ずつ処理する）
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_names = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_names[idx])
        image = Image.open(img_name).convert('L')  # 白黒画像として開く
        if self.transform:
            image = self.transform(image)
        return image

# データセットの読み込み
image_folder = 'make_fig/fig_data'  # 画像のディレクトリパス
dataset = CustomDataset(image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# モデルのインスタンス化
model = AutoEncoder()

# 学習済みのモデルを読み込む
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # 推論モードに設定

# 推論を行う
with torch.no_grad():  # 勾配計算を不要にする
    for i, data in enumerate(dataloader):
        # 1枚ずつ画像を取得
        output = model(data)

        # 出力を画像に変換して保存（例: 'output_1.png'）
        output_img = output.squeeze().cpu().numpy()  # バッチサイズを削除
        output_img = (output_img * 255).astype('uint8')  # ピクセル値を255にスケール
        output_image = Image.fromarray(output_img[0], mode='L')
        output_image.save(f'output_{i+1}.png')
