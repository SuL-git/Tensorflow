import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
#输入数据的占位符
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
#w = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))
#卷积网络对图像进行分类需将向量还原为28*28的图片形式
x_image = tf.reshape(x,[-1,28,28,1])

def weight_variable(shape):#返回给定形状的变量并自动截断正态分布初始化
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):#返回给定形状的变量，初始化所有的值是0.1
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
#第一层卷积
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
#池化
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
#池化
h_pool2 = max_pool_2x2(h_conv2)
#第一个全连接层
w_fc1 = weight_variable([7 * 7 * 64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)#连接去除的概率，训练时是0.5，测试时是1
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#第二个全连接层
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop,w_fc2) + b_fc2
#定义损失函数，交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#预测精度
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化所有变量
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#不断迭代
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})
#打印最终精度
print("test accuracy %g"% accuracy.eval(feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
#写入tensorboard
writer = tf.summary.FileWriter('C://Users//ko936//Desktop//tensorflow//tensorboard//mycnngraph',sess.graph)
writer.close
        