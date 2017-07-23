# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def load_CIFAR10(file):
    # get the training data
    dataTrain = []
    labelTrain = []
    dic = unpickle(file)
    for item in dic['data']:
        dataTrain.append(item)
    for item in dic['labels']:
        labelTrain.append(item)

    return dataTrain, labelTrain


LOGDIR = 'cifarV'
NAME_TO_VISUALISE_VARIABLE = 'cifarembedding'

path_for_cifar_sprites = os.path.join(LOGDIR, 'cifardigits.png')
path_for_cifar_metadata = os.path.join(LOGDIR, 'metadata.tsv')

# Load data sets
Xtr, Ytr = load_CIFAR10(
    '/Users/meidongyang/dataset/cifar-10-batches-py/data_batch_1')
Xtr = np.asarray(Xtr)
Ytr = np.asarray(Ytr)
Xtr = Xtr[:500, ]
Ytr = Ytr[:500, ]
print Xtr.shape, Ytr.shape
print Xtr
print Ytr
# # flatten out all images to be one-dimensional
# # Xtr_rows becomes 50000 x 3072
# Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
# # Xte_rows becomes 10000 x 3072
# Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

# 2. 建立 embeddings，最主要的就是你要知道想可视化查看的 variable 的名字：
embedding_var = tf.Variable(Xtr, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOGDIR)

# 3. 建立embedding projector：这一步很重要，要指定想要可视化的 variable，metadata 文件的位置
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the metadata
embedding.metadata_path = path_for_cifar_metadata  # 'metadata.tsv'

# Specify where you find the sprite (we will create this later)
embedding.sprite.image_path = path_for_cifar_sprites  # 'cifardigits.png'
embedding.sprite.single_image_dim.extend([32, 32])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

# 4. 保存：
# Tensorboard 会从保存的图形中加载保存的变量，所以初始化 session 和变量，并将其保存在 logdir 中
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOGDIR, 'model.ckpt'), 1)


# 5. 定义 helper functions：
# **create_sprite_image:** 将 sprits 整齐地对齐在方形画布上
# **vector_to_matrix_cifar:** 将 cifar 的 vector 数据形式转化为 images
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument.
    Images should be count x width x height"""
    if isinstance(images, list):
        np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # ceil()方法返回x的值上限 - 不小于x的最小整数。
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h: (i + 1) * img_h,
                            j * img_w: (j + 1) * img_w] = this_img
    return spriteimage


def vector_to_matrix_cifar(cifar_digits):
    """Reshapes normal cifar digit (batch,32*32) to matrix (batch,32,32)"""
    return np.reshape(cifar_digits, (-1, 32, 32))


# 6. 保存 sprite image：
# 将 vector 转换为 images，创建并保存 sprite image。
to_visualise = Xtr
to_visualise = vector_to_matrix_cifar(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(path_for_cifar_sprites, sprite_image, cmap=plt.cm.Spectral)
plt.imshow(sprite_image, cmap=plt.cm.Spectral)

# 7. 保存 metadata:
# 将数据写入 metadata，因为如果想在可视化时看到不同数字用不同颜色表示，
# 需要知道每个 image 的标签，在这个 metadata 文件中有这样两列：”Index” , “Label”
with open(path_for_cifar_metadata, 'w') as f:
    f.write('Index\tLabel\n')
    for index, label in enumerate(Ytr):
        f.write("%d\t%d\n" % (index, label))
