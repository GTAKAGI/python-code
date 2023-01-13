import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io

"""
##label imageを見たいときはコメントアウト外してください

name = scipy.io.loadmat('label/label_100.mat')
print(name)
print(name['name'].shape)

img = name['name']
np.set_printoptions(threshold=np.inf)
print(img)

plt.imshow(img)
plt.show()
"""
#image　を見たいときはコメントアウトを外してください

# number = input('何番読み込む？')
# path = r'C:\Users\josep\Desktop\SNN\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder/image/image_'+number
# name = scipy.io.loadmat(path)
# print(name)

# # print(type(name)) 
# # print(name.keys()) #key(value)
# # print(name.values()) 
# # print(name.items())
# # print(name['time_data'])
# print(type(name['time_data']))
# print("============")
# print(name['time_data'].shape)
# print(name['time_data'][:,:,1]) #1層目の64×64ピクセルをすべて表示
# img = name['time_data'][:,:,1]
# np.set_printoptions(threshold=np.inf) #全表示(省略しない)
# print(img)

# N = 11
# fig, ax = plt.subplots()
# def update(i):
#     img = name['time_data'][:,:,i]
    
#     plt.clf()
    
#     plt.imshow(img,cmap='gray')
# ani = animation.FuncAnimation(fig, update, np.arange(0,  N), interval=200)  # 代入しないと消される
# ani.save(str(path)+'.gif',writer='imagemagick')
# plt.show()

##LiDARのgifをフレームレートに分けて画像で取得
from pathlib import Path
from PIL import Image, ImageSequence
import cv2
number = input('何番読み込む？')
# 分割したいアニメーション GIF 画像
IMAGE_PATH = r'C:\Users\josep\Desktop\SNN\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder/image/image_'+number+'.gif'
# 分割した画像の出力先ディレクトリ
DESTINATION = r'C:\Users\josep\Desktop\SNN\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder/splitted_img/splitted_img_'+number
# 現在の状況を標準出力に表示するかどうか
DEBUG_MODE = True

def main():
    frames = get_frames(IMAGE_PATH)
    write_frames(frames, IMAGE_PATH, DESTINATION)

def get_frames(path):
    '''パスで指定されたファイルのフレーム一覧を取得する
    '''
    im = Image.open(path)
    return (frame.copy() for frame in ImageSequence.Iterator(im))

def write_frames(frames, name_original, destination):
    '''フレームを別個の画像ファイルとして保存する
    '''
    path = Path(name_original)

    stem = path.stem
    extension = path.suffix

    # 出力先のディレクトリが存在しなければ作成しておく
    dir_dest = Path(destination)
    if not dir_dest.is_dir():
        dir_dest.mkdir(0o700)
        if DEBUG_MODE:
            print('Destionation directory is created: "{}".'.format(destination))

    for i, f in enumerate(frames):
        name = '{}/{}-{}{}'.format(destination, stem, i + 1, extension)
        f.save(name)
        if DEBUG_MODE:
            print('A frame is saved as "{}".'.format(name))
    print(frames)

if __name__ == '__main__':
    main()

# import cv2
# import glob
# import numpy as np
# from PIL import Image
# from natsort import natsorted
# from matplotlib import pyplot as plt
# # タイル状に pm × pm 枚配置
# height = 2
# width = 5
# # 所定のフォルダ内にあるjpgファイルを連続で読み込んでリスト化する
# files = glob.glob(DESTINATION)
# # 空の入れ物（リスト）を準備
# d = []
# # ファイル名が数字の場合、natsortedで
# # 自然順（ファイル番号の小さい順）に1枚づつ読み込まれる
# for i in natsorted(files):
#     img = Image.open(i)
#     img = np.asarray(img)
#     #img = cv2.resize(img, (300, 300), cv2.INTER_LANCZOS4)
#     d.append(img)
# # タイル状に画像を一覧表示
# fig, ax = plt.subplots(height, width, figsize=(10, 10))
# fig.subplots_adjust(hspace=0, wspace=0)
# for i in range(width):
#     for j in range(height):
#         ax[i, j].xaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].yaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].imshow(d[pm*i+j], cmap="bone")
plt.show()

# def concat_tile(im_list_2d):
#     return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

# im1_s = cv2.resize(im1, dsize=(0, 0), fx=0.5, fy=0.5)
# im_tile = concat_tile([[im1_s, im1_s, im1_s, im1_s, im1_s],
#                        [im1_s, im1_s, im1_s, im1_s, im1_s],
#                        ])
# cv2.imwrite('data/dst/opencv_concat_tile.jpg', im_tile)