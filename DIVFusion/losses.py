# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import time
import cv2

# VGG에서 제공하는 상수, 이전 VGG 훈련을 통해 정규화되었으므로 지금도 동일한 작업이 필요합니다.
VGG_MEAN = [103.939, 116.779, 123.68]  # rgb 세 채널의 평균값


class VGGNet():
    '''
    vgg16 네트워크 구조 생성
    모델에서 매개변수 로드
    '''

    def __init__(self, data_dict):
        '''
        vgg16 모델 전달
        :param data_dict: vgg16.npy (딕셔너리 타입)
        '''
        self.data_dict = data_dict

    def get_conv_filter(self, name):
        '''
        해당 이름의 컨볼루션 레이어 얻기
        :param name: 컨볼루션 레이어 이름
        :return: 해당 컨볼루션 레이어 출력
        '''
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        '''
        이름이 name인 완전 연결 레이어의 가중치 얻기
        :param name: 연결 레이어 이름
        :return: 해당 레이어 가중치
        '''
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        '''
        이름이 name인 완전 연결 레이어의 편향 얻기
        :param name: 연결 레이어 이름
        :return: 해당 레이어 편향
        '''
        return tf.constant(self.data_dict[name][1], name='bias')

    def conv_layer(self, x, name):
        '''
        컨볼루션 레이어 생성
        :param x:
        :param name:
        :return:
        '''
        # 계산 그래프 모델을 작성할 때, 일부 필요한 name_scope를 추가하는 것이 좋은 코딩 규칙입니다.
        # 이는 이름 충돌을 방지하고, 계산 그래프를 시각화할 때 더 명확하게 합니다.
        with tf.name_scope(name):
            # w와 b 얻기
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)

            # 컨볼루션 계산 수행
            h = tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding='SAME')
            '''
            현재 w와 b는 외부에서 전달되므로 tf.nn.conv2d()를 사용합니다.
            tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu = None, name = None) 매개변수 설명:
            input 입력 텐서, 형식 [batch, height, width, channel]
            filter 컨볼루션 커널 [filter_height, filter_width, in_channels, out_channels] 
                각각: 컨볼루션 커널 높이, 컨볼루션 커널 너비, 입력 채널 수, 출력 채널 수
            strides 스트라이드, 컨볼루션 시 이미지의 각 차원에서의 스트라이드, 길이 4
            padding 매개변수는 "SAME" 또는 "VALID"를 선택할 수 있습니다.

            '''
            # 편향 추가
            h = tf.nn.bias_add(h, conv_b)
            # 활성화 함수 사용
            h = tf.nn.relu(h)
            return h

    def pooling_layer(self, x, name):
        '''
        풀링 레이어 생성
        :param x: 입력 텐서
        :param name: 풀링 레이어 이름
        :return: 텐서
        '''
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],  # 커널 매개변수, 주의: 모두 4차원
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name
                              )

    def fc_layer(self, x, name, activation=tf.nn.relu):
        '''
        완전 연결 레이어 생성
        :param x: 입력 텐서
        :param name: 완전 연결 레이어 이름
        :param activation: 활성화 함수 이름
        :return: 출력 텐서
        '''
        with tf.name_scope(name, activation):
            # 완전 연결 레이어의 w와 b 얻기
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            # 행렬 곱셈 계산
            h = tf.matmul(x, fc_w)
            # 편향 추가
            h = tf.nn.bias_add(h, fc_b)
            # 마지막 레이어에는 활성화 함수 relu가 없으므로 여기서 판단해야 합니다.
            if activation is None:
                return h
            else:
                return activation(h)

    def flatten_layer(self, x, name):
        '''
        평탄화
        :param x: input_tensor
        :param name:
        :return: 2차원 행렬
        '''
        with tf.name_scope(name):
            # [batch_size, image_width, image_height, channel]
            x_shape = x.get_shape().as_list()
            # 마지막 세 차원을 합친 크기 계산
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            # 2차원 행렬 형성
            x = tf.reshape(x, [-1, dim])
            return x

    def build(self, x_rgb):
        '''
        vgg16 네트워크 생성
        :param x_rgb: [1, 224, 224, 3]
        :return:
        '''
        start_time = time.time()
        # print('模型开始创建……')
        # 将输入图像进行处理，将每个通道减去均值
        # r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)
        '''
        tf.split(value, num_or_size_split, axis=0)用法：
        value:输入的Tensor
        num_or_size_split:有两种用法：
            1.直接传入一个整数，代表会被切成几个张量，切割的维度有axis指定
            2.传入一个向量，向量长度就是被切的份数。传入向量的好处在于，可以指定每一份有多少元素
        axis, 指定从哪一个维度切割
        因此，上一句的意思就是从第4维切分，分为3份，每一份只有1个元素
        '''
        # 将 处理后的通道再次合并起来
        # x_bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)

        #        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 开始构建卷积层
        # vgg16 的网络结构
        # 第一层：2个卷积层 1个pooling层
        # 第二层：2个卷积层 1个pooling层
        # 第三层：3个卷积层 1个pooling层
        # 第四层：3个卷积层 1个pooling层
        # 第五层：3个卷积层 1个pooling层
        # 第六层： 全连接
        # 第七层： 全连接
        # 第八层： 全连接

        # 这些变量名称不能乱取，必须要和vgg16模型保持一致,
        # 另外，将这些卷积层用self.的形式，方便以后取用方便
        self.conv1_1 = self.conv_layer(x_rgb, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        # print('创建模型结束：%4ds' % (time.time() - start_time))



def get_row_col(num_pic):
    '''
    计算行列的值
    :param num_pic: 特征图的数量
    :return:
    '''
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

# def visualize_feature_map(feature_batch):
#     '''
#     创建特征子图，创建叠加后的特征图
#     :param feature_batch: 一个卷积层所有特征图
#     :return:
#     '''
#     feature_map = np.squeeze(feature_batch, axis=0)
#
#     feature_map_combination = []
#     plt.figure(figsize=(8, 7))
#
#     # 取出 featurn map 的数量，因为特征图数量很多，这里直接手动指定了。
#     #num_pic = feature_map.shape[2]
#
#     row, col = get_row_col(25)
#     # 将 每一层卷积的特征图，拼接层 5 × 5
#     for i in range(0, 25):
#         feature_map_split = feature_map[:, :, i]
#         feature_map_combination.append(feature_map_split)
#         plt.subplot(row, col, i+1)
#         plt.imshow(feature_map_split)
#         plt.axis('off')
#
#     #plt.savefig('./mao_feature/feature_map2.png') # 保存图像到本地
#     plt.show()


def visualize_feature_map_sum(feature_batch):
    '''
    将每张子图进行相加
    :param feature_batch:
    :return:
    '''
    feature_map = np.squeeze(feature_batch, axis=0)

    feature_map_combination = []

    # 取出 featurn map 的数量
    num_pic = feature_map.shape[2]

    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)

    # 按照特征图 进行 叠加代码

    feature_map_sum = sum(one for one in feature_map_combination)
    return feature_map_sum
    # plt.imshow(feature_map_sum)
    # #plt.savefig('./mao_feature/feature_map_sum2.png') # 保存图像到本地
    # plt.show()


def get_feature(image, layer):
    # content = tf.placeholder(tf.float32, shape=[1, 450, 620, 3])
    vgg16_npy_pyth = 'vgg16.npy'
    # content = tf.concat([image, image, image], axis=-1)
    content = image
    # 载入模型， 注意：在python3中，需要添加一句： encoding='latin1'
    data_dict = np.load(vgg16_npy_pyth, encoding='latin1', allow_pickle=True).item()

    # 创建图像的 vgg 对象
    vgg_for_content = VGGNet(data_dict)

    # 创建 每个 神经网络
    vgg_for_content.build(content)

    content_features = [vgg_for_content.conv1_2,
                        vgg_for_content.conv2_2,
                        vgg_for_content.conv3_3,
                        vgg_for_content.conv4_3,
                        vgg_for_content.conv5_3,
                        ]

    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     content_features = sess.run([content_features],
    #                                 feed_dict={
    #                                     content: image
    #                                 })

    conv1 = content_features[0]
    conv2 = content_features[1]
    conv3 = content_features[2]
    conv4 = content_features[3]
    conv5 = content_features[4]
    if layer == 'conv1':
        return conv1
    if layer == 'conv2':
        return conv2
    if layer == 'conv3':
        return conv3
    if layer == 'conv4':
        return conv4
    if layer == 'conv5':
        return conv5
    # # 查看 每个 特征 子图
    # visualize_feature_map(conv3)
    #
    # # 查看 叠加后的 特征图
    # visualize_feature_map_sum(conv3)

def Perceptual_Loss(Orignal_image, Generate_image):
    orignal_conv3_feature = get_feature(Orignal_image, 'conv3')
    orignal_conv4_feature = get_feature(Orignal_image, 'conv4')
    Generate_conv3_feature = get_feature(Generate_image, 'conv3')
    Generate_conv4_feature = get_feature(Generate_image, 'conv4')
    conv3_LOSS = tf.reduce_mean(tf.abs(orignal_conv3_feature - Generate_conv3_feature))
    conv4_LOSS = tf.reduce_mean(tf.abs(orignal_conv4_feature - Generate_conv4_feature))
    Perceptualloss = conv3_LOSS + conv4_LOSS
    return Perceptualloss


def TV_LOSS(batchimg):
    TV_norm = tf.reduce_sum(tf.abs(batchimg), axis=[1, 2, 3])
    E = tf.reduce_mean(TV_norm)
    return E








