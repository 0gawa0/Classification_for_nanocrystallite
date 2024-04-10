import numpy as np
import os

from glob import glob
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import ImageFile
from tensorflow.keras.models import Model

#分類クラス
classes = [
           'shima', 'shimanashi','taisyougai'
           ]

#データパス(datasetには縞模様のディレクトリ、模様なしのディレクトリ、対象外のディレクトリ、全てのデータのディレクトリの４つで構成するものとする)
data_basedir = f'./data_set/test'
# セーブパス
save_dir = f'./np_file'

# def train_images():# 訓練画像のジェネレータ
#     for (i,x) in enumerate(classes):
#         images = glob('{}/{}/*.jpg'.format(data_basedir, x))
#         for im in sorted(images):
#             yield im,i

listAll = []
listIndex = []
total = 0
filecnt = 0
for folder in classes:
    # あり、なしそれぞれのフォルダーに入っている画像ファイル数を計算、ファイル名を記録
	flist = os.listdir( f'{data_basedir}/{folder}') # using os list directory
	length = len(flist)
    # [0,0,0,0, ... ,1,1,1,1]　多分ここでラベルを付与
	for i in range(length):
		listIndex.append(filecnt)
	total += len(flist)
	listAll.append(flist)
	filecnt += 1
print("画像ファイル数 = ",total)


# 処理対象の訓練用の全画像をリストアップ
# image_list (flat_list)に処理対象の画像ファイル名（ファイルパス）が入っていると仮定 
# ImageFile.LOAD_TRUNCATED_IMAGES = True
image_list = [item for sublist in listAll for item in sublist]
training_list = []

# モデル生成
base_model = VGG19(weights='imagenet')
pp_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
pp_model.summary()

for i in range(len(image_list)):
    if i % 400 == 0:
        print("Processing ...",(i+1)," data...")
    img_path = data_basedir + "/all/" + image_list[i]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x) 
    fc2_ft = pp_model.predict(x) 
    training_list.append(fc2_ft)

fc2_list = np.asarray(training_list) 
fc2_training = fc2_list.reshape([len(fc2_list),4096])
print(fc2_training.shape)

os.chdir(save_dir)
np.save(f'test.npy',fc2_training) # 結果をNumPy形式のファイルに保存
