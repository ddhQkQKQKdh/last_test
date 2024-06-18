import numpy as np
import struct
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# Ubyte 파일을 읽는 함수 정의
def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images

def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# 데이터 읽기 (Fashion MNIST)
X_train = load_mnist_images('train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

# 데이터 전처리
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 레이블 전처리
t_train = y_train
t_test = y_test

# 데이터 줄이기 (학습 속도를 위해 일부 데이터만 사용)
X_train, t_train = X_train[:6000], t_train[:6000]  # 학습 데이터를 6000개로 설정
X_test, t_test = X_test[:3000], t_test[:3000]  # 테스트 데이터를 3000개로 설정

max_epochs = 30  # 에포크 수를 50으로 설정

# SimpleConvNet 모델 설정
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 필터 수를 32로 설정
                        hidden_size=128,  # 히든 레이어의 뉴런 수를 128로 설정
                        output_size=10, weight_init_std=0.01)

# Trainer 설정 및 훈련
trainer = Trainer(network, X_train, t_train, X_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,  # 미니 배치 크기를 100으로 설정
                  optimizer='Adam', optimizer_param={'lr': 0.001},  # 학습률을 0.001로 설정
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
plt.show()
