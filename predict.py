from matplotlib import pylab
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score
from matplotlib import font_manager
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
import numpy as np
import itertools
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import warnings
import tensorflow.compat.v1 as tf
import keras
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #40%
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

#分類クラス
classes = [
           'shima', 'shimanashi','taisyougai'
           ]

#検証データのパス
data_basedir = f'./data_set/test'

save_model_dir = './model'
save_graph_dir = './graph'
save_text_dir = './text'

def train_images():# 訓練画像のジェネレータ
    for (i,x) in enumerate(classes):
        images = glob('{}/{}/*.jpg'.format(data_basedir, x))
        for im in sorted(images):
            yield im,i

val_x = np.load(f'./np_file/test.npy') #VGGを通して得た特徴量

img_labels = []
for im, ell in train_images():
    img_labels.append(ell)
img_labels = np.array(img_labels)

print(img_labels.shape)

val_y = utils.to_categorical(img_labels, 3)

# ニューラルネットワークのモデル構築
font_path='./font/ipaexg.ttf' 
font_prop = font_manager.FontProperties(fname=font_path)
font_prop.set_style('normal')
font_prop.set_weight('light')
font_prop.set_size('12')
fp2 = font_prop.copy()
fp2.set_size('25')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Oranges):

    plt.figure(figsize=(14,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=25, font_properties=fp2)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=23)
    plt.yticks(tick_marks, classes,fontsize=23)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('a')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center", fontsize=50,
                color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('予測値',fontsize=35,font_properties=fp2)
    plt.xlabel('真値',fontsize=35,font_properties=fp2)

val_score = []
ave_val_score = []
f1_ave = []

#5分割交差検証を行っているため5個のモデルでそれぞれ精度を確認
for i in range(5):
  i += 1
  #モデルのロード
  model = load_model(f'{save_model_dir}/model_parameter{i}.h5') # 重みやバイアスといったパラメータを読み込む
  val_loss, val_acc = model.evaluate(val_x, val_y, verbose=1)
  print(val_acc)
  val_score.append(val_acc)
  y_pred = model.predict(val_x) 
  y_pred_classes = np.argmax(y_pred, axis=1) 
  y_real_classes = np.argmax(val_y, axis=1) 
  cm = confusion_matrix(y_pred_classes, y_real_classes)
  f_score = f1_score(y_real_classes, y_pred_classes, average='micro')
  print('f1 = ',f_score)
  f1_ave.append(f_score)
  classes = ['cat', 'dog', 'house']
  plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix')
  plt.savefig(f'{save_graph_dir}/val_result_matrix{i}.png')


#検証結果の保存
from pathlib import Path
p0 = Path(f'{save_text_dir}/val_result.txt')
p0.write_text(f'acc = {val_score}\n average = {np.mean(val_score)} \n f1 = {f1_ave} \n f1 average = {np.mean(f1_ave)} \n variance = {np.var(val_score)*10000}')

