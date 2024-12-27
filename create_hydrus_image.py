from PIL import Image, ImageDraw
import math
import numpy as np

# 画像のサイズと棒・円のパラメータを設定
image_width = 600
image_height = 600

bar_width = 60  # 棒の長さ
bar_height = 2  # 棒の太さ
bar_color = (0, 0, 0)  # 黒色 (R, G, B)

circle_radius = 20  # 円の半径
circle_color = (0, 0, 0)  # 黒色 (R, G, B)

background_color = (255, 255, 255)  # 白色 (R, G, B)

def make_shapes(initial_angle, joint_angles, idx):
    whole_angles = initial_angle
    # 画像を作成
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    
    # 最初の棒の角度をラジアンに変換
    initial_angle_rad = math.radians(initial_angle)
    
    # 最初の棒の位置 (中心からスタート)
    start_x = image_width // 2
    start_y = image_height // 2
    
    # 最初の棒の終端を計算 (角度を考慮)
    end_x = start_x + bar_width * math.cos(initial_angle_rad)
    end_y = start_y + bar_width * math.sin(initial_angle_rad)
    
    # 棒を描画
    draw.line(
        [(start_x, start_y), (end_x, end_y)],
        fill=bar_color,
        width=bar_height
    )
    
    # 棒の中心を計算
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    
    # 円を描画（棒の中心に配置）
    draw.ellipse(
        [
            mid_x - circle_radius, mid_y - circle_radius,
            mid_x + circle_radius, mid_y + circle_radius
        ],
        fill=circle_color
    )
    
    # **for文で角度ごとに追加描画**
    for angle in joint_angles:
        whole_angles += angle
        # 角度をラジアンに変換
        angle_rad = math.radians(whole_angles)
        
        # 次の棒の始点は現在の棒の終端
        start_x = end_x
        start_y = end_y
        
        # 次の棒の終端を計算 (角度を考慮)
        end_x = start_x + bar_width * math.cos(angle_rad)
        end_y = start_y + bar_width * math.sin(angle_rad)
        
        # 棒を描画
        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=bar_color,
            width=bar_height
        )
        
        # 次の棒の中心を計算
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # 円を描画（棒の中心に配置）
        draw.ellipse(
            [
                mid_x - circle_radius, mid_y - circle_radius,
                mid_x + circle_radius, mid_y + circle_radius
            ],
            fill=circle_color
        )
    
    # 画像内の黒い部分（円と棒）の中心を計算
    pixels = image.load()  # 画像のピクセルデータを取得
    
    # 黒い部分のピクセルの合計
    total_x = 0
    total_y = 0
    count = 0
    
    # 黒い部分のピクセルを探して、その中心を計算
    for x in range(image_width):
        for y in range(image_height):
            if pixels[x, y] == bar_color:  # 黒いピクセル（棒の部分）
                total_x += x
                total_y += y
                count += 1
            elif pixels[x, y] == circle_color:  # 黒いピクセル（円の部分）
                total_x += x
                total_y += y
                count += 1
    
    if count > 0:
        black_center_x = total_x // count  # 黒い部分の中心X
        black_center_y = total_y // count  # 黒い部分の中心Y
    else:
        black_center_x = image_width // 2
        black_center_y = image_height // 2
    
    # 新しい画像を作成し、黒い部分をその中心に合わせて移動
    new_image = Image.new("RGB", (image_width, image_height), background_color)
    shift_x = image_width // 2 - black_center_x
    shift_y = image_height // 2 - black_center_y
    
    # 黒い部分を新しい画像に配置
    new_image.paste(image, (shift_x, shift_y))
    
    # 画像を縮小（20x20ピクセルに）
    small_image = new_image.resize((112, 112))
    
    # 縮小した画像を保存
    small_image.save("initial_angle"+str(int(initial_angle)).zfill(3)+"_"+str(idx).zfill(3) +".png")


num_init_angles = 13
num_joint_steps = 7

for init_angle in np.linspace(0, 360, num_init_angles):
    idx = 0
    for angle_1 in np.linspace(-90, 90, num_joint_steps):
        for angle_2 in np.linspace(-90, 90, num_joint_steps):
            for angle_3 in np.linspace(-90, 90, num_joint_steps):
                joint_angles = [angle_1, angle_2, angle_3]     
                make_shapes(init_angle, joint_angles, idx)
                idx+=1;
