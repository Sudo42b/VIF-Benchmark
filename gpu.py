import tensorflow as tf
import time
# 세션을 열고 GPU에서 연산을 실행
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 필요할 때만 메모리 할당
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 사용할 메모리 비율 제한
# TensorFlow 1.x의 경우, Eager Execution을 비활성화해야 합니다.
tf.compat.v1.disable_eager_execution()

# 임의의 큰 행렬을 생성
matrix1 = tf.random.uniform([3000, 3000])
matrix2 = tf.random.uniform([3000, 3000])

# 행렬 곱셈 연산 정의
product = tf.matmul(matrix1, matrix2)



with tf.compat.v1.Session(config=config) as sess:
    start_time = time.time()
    result = sess.run(product)
    end_time = time.time()

    print("GPU에서 연산 완료")
    print("연산 소요 시간: {:.4f} 초".format(end_time - start_time))

#4. Tensorflow1 수행 테스트
# import tensorflow as tf
print(tf.__version__)
# 1. check if tensorflow gpu is installed
print(tf.test.gpu_device_name())
# 2. tensorflow gpu test
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))