from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from tensorflow.keras import utils
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
from glob import glob
from matplotlib import pylab
from matplotlib import font_manager
import numpy as np
import warnings
import tensorflow.compat.v1 as tf
import tensorflow.keras
import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects

#プロット設定
font_path='./font/ipaexg.ttf' 
font_prop = font_manager.FontProperties(fname=font_path)
font_prop.set_style('normal')
font_prop.set_weight('light')
font_prop.set_size('12')
fp2 = font_prop.copy()
fp2.set_size('25')

# ハードウェアリソースの設定など
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

#分類クラス
classes = [
           'shima', 'shimanashi','taisyougai'
           ]

#学習データのパス
data_basedir = f'./data_set/train'

save_model_dir = './model'
save_graph_dir = './graph'

def train_images():# 訓練画像のジェネレータ
    for (i,x) in enumerate(classes):
        images = glob('{}/{}/*.jpg'.format(data_basedir, x))
        for im in sorted(images):
            yield im,i

#混同行列
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='混合行列',
                          cmap=plt.cm.Oranges):

    plt.figure(figsize=(14,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=25, font_properties=fp2)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=23)
    plt.yticks(tick_marks, classes,fontsize=23)

    # 正規化するか
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


# ここからKerasによるニューラルネットワークによる深層学習スタート
# 学習データのパスに移動

#学 習データのnumpy配列読み込み
fc2_training = np.load(f'./np_file/from_imgs.npy')

img_labels = []
for im, ell in train_images():
    img_labels.append(ell)

img_labels = np.array(img_labels)
nnum = 1
val_score = []
ave_val_score = []
f1_ave = []
NB_CLASSES = 3
NB_INPUTS = 4096
NB_HIDDEN = 256
NB_BATCH = 128
NB_EPOCHS = 100

kf = StratifiedKFold(n_splits=5, shuffle=True)
# 5分割交差検証
for train_index, val_index in kf.split(fc2_training, img_labels):

  # Xは画像データ、Yはラベル
  X_tra, X_tes = fc2_training[train_index], fc2_training[val_index]
  Y_tra, Y_tes = img_labels[train_index], img_labels[val_index]

  print('X_tra', len(X_tra))
  print('X_tes', len(X_tes))

  #モデルの構築
  a_model = Sequential()
  a_model.add(Dense(NB_HIDDEN,input_dim=NB_INPUTS,activation='relu'))
  a_model.add(BatchNormalization())
  a_model.add(Dropout(0.3))
  a_model.add(Dense(NB_CLASSES,activation='softmax'))
  a_compile = a_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) 
  my_label = utils.to_categorical(Y_tra, NB_CLASSES)
  my_test_label = utils.to_categorical(Y_tes, NB_CLASSES)
  callbacks_list = [
      tf.keras.callbacks.EarlyStopping( monitor='loss',patience=5, ),  # 5エポック以上改善なければストップ
  ]

  # 学習開始
  a_fit = a_model.fit(X_tra, my_label, 
                    epochs=NB_EPOCHS, 
                    batch_size=NB_BATCH,
                    callbacks = callbacks_list,
                    validation_data=(X_tes, my_test_label))

  y_pred = a_model.predict(X_tes) 
  y_pred_classes = np.argmax(y_pred, axis=1) 
  y_real_classes = np.argmax(my_test_label, axis=1) 
  cm = confusion_matrix(y_pred_classes, y_real_classes)

  valid_score = a_fit.history['val_accuracy'][-1]
  ave_val_score.append(valid_score) 

  f_score = f1_score(y_real_classes, y_pred_classes, average='micro')
  print(f_score)
  f1_ave.append(f_score)

  #モデルの保存
  a_model.save(f'{save_model_dir}/model_parameter{nnum}.h5')



  #混同行列の保存
  classes = ['cat', 'dog', 'house']
  plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix{0}'.format(nnum))
  plt.savefig(f'{save_graph_dir}/{nnum}_matrix.jpg')

  plt.figure(figsize=(14,10))
  plt.plot(a_fit.history['accuracy'],
           color='b', 
           linestyle='-', 
           linewidth=3, 
           path_effects=[path_effects.SimpleLineShadow(),
                         path_effects.Normal()])
  plt.plot(a_fit.history['val_accuracy'], 
           color='r', 
           linestyle='--',
           linewidth=3,
           path_effects=[path_effects.SimpleLineShadow(),
                         path_effects.Normal()])

  plt.tick_params(labelsize=18)

  plt.title('エポック-精度グラフ',fontsize=30,font_properties=fp2)
  plt.ylabel('精度',fontsize=25,font_properties=fp2)
  plt.xlabel('エポック',fontsize=25,font_properties=fp2)
  plt.legend(['訓練', 'テスト'], loc='best', fontsize=20, prop=fp2)

  #精度グラフの保存
  plt.savefig(f'{save_graph_dir}/{nnum}_acc.jpg')

  plt.figure(figsize=(14,10))

  plt.plot(a_fit.history['loss'], 
           color='b', 
           linestyle='-', 
           linewidth=3, 
           path_effects=[path_effects.SimpleLineShadow(),
                         path_effects.Normal()])
  plt.plot(a_fit.history['val_loss'], 
           color='r', 
           linestyle='--',
           linewidth=3,
           path_effects=[path_effects.SimpleLineShadow(),
                         path_effects.Normal()])

  plt.tick_params(labelsize=18)

  plt.title('エポック-損失グラフ',fontsize=30,font_properties=fp2)
  plt.ylabel('損失',fontsize=25,font_properties=fp2)
  plt.xlabel('エポック',fontsize=25,font_properties=fp2)
  plt.legend(['訓練', 'テスト'], loc='best', fontsize=20, prop=fp2)

  #損失グラフの保存
  plt.savefig(f'{save_graph_dir}/{nnum}_loss.jpg')

  nnum += 1
