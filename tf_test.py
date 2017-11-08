import tensorflow as tf
"""
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")
result = a+b
print(result)

print( a.graph is tf.get_default_graph())

g = tf.Graph()
with g.device('/gpu:0'):
	result = a+b
	print(result)

c = tf.constant([1, 2], name = "c", dtype=tf.float32)
result = a+b
print(c)

print("session........")
sess = tf.Session()
with sess.as_default():
	print(result.eval())

sess = tf.Session()
print(sess.run(result))
print(result.eval(session=sess))
sess.close()

sess = tf.InteractiveSession()
print(result.eval())
sess.close()

print("nnet.....nnet....")
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed = 1))
x = tf.constant([[0.7, 0.9]])
print(x)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()

print("nnet place hold 1..........")
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed = 1))
x = tf.placeholder(tf.float32, shape=(1,2), name="input")
print(x)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y, feed_dict={x:[[0.7, 0.9]]}))
sess.close()


print("nnet place hold 2..........")
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed = 1))
x = tf.placeholder(tf.float32, shape=(3,2), name="input")
print(x)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y, feed_dict={x:[[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
sess.close()

"""
print("nnet all..........")
from numpy.random import RandomState
batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed = 1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="x-input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

cross_entropy = -tf.reduce_mean( y_ * tf.log(tf.clip_by_value( y, 1e-10, 1.0 )))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X=rdm.rand(dataset_size, 2)
Y=[[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print sess.run(w1)
	print sess.run(w2)

	STEPS = 5000
	for i in range(STEPS):
		start = ( i * batch_size ) % dataset_size
		ed = min(start+batch_size, dataset_size)
		#sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		sess.run(train_step, feed_dict={x: X[start:128], y_: Y[start:128]})

		if i %1000 == 0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
			print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))

	print sess.run(w1)
	print sess.run(w2)

