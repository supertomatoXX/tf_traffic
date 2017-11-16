#coding=utf-8
# Import `tensorflow`
import tensorflow as tf
from load import*
import random
import matplotlib.pyplot as plt

# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(tf.int32, [None])


#然后构建你的网络。首先使用 flatten() 函数展平输入，
# 其会给你一个形状为 [None, 784] 的数组，而不是 [None, 28, 28]——这是你的灰度图像的形状。
# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer构建一个全连接层，其可以生成大小为 [None, 62] 的 logits。logits 是运行在早期层未缩放的输出上的函数，其使用相对比例来了解单位是否是线性的。
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
#定义损失函数了。sparse_softmax_cross_entropy_with_logits()，其可以计算 logits 和标签之间的稀疏 softmax 交叉熵。回归(regression)被用于预测连续值，而分类(classification)则被用于预测离散值或数据点的类别。你可以使用 reduce_mean() 来包裹这个函数，它可以计算一个张量的维度上各个元素的均值。
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
#                                                                     logits=logits))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                     labels=y))


# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''print("images_flat: ", images_flat)

print("logits: ", logits)

print("loss: ", loss)

print("predicted_labels: ", correct_pred)
'''
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    #print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", accuracy_val)
    #print('DONE WITH EPOCH')


# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]
# Run the "predicted_labels" op.
predicted = sess.run(correct_pred,
		                        {x: sample_images})
print(sample_labels)
print(predicted)


# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()
