import input_data
import sys
import tensorflow as tf

mnist = input_data.read_data_sets("data/", one_hot=True)

def inference(x):
    init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', [784, 10], initializer=init)
    b = tf.get_variable('b', [10], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    W_hist = tf.summary.histogram('wights', W)
    b_hsit = tf.summary.histogram('biases', b)
    y_hist = tf.summary.histogram('output', output)
    return output

def loss(output, y):
    dot_product = y * tf.log(output)

    # Reduction along axis 0 collapses each column into a
    # single value, whereas reduction along axis 1 collapses
    # each row into a single value. In general, reduction along
    # axis i collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
    loss = tf.reduce_mean(xentropy)
    tf.summary.scalar('losss', loss)
    return loss

def training(cost, learning_rate, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1),
                                  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))
    tf.summary.scalar('validation accuracy', accuracy)
    return accuracy

def main(args):
    learning_rate = 0.01
    nrof_training_epoch = 300
    batch_size = 100
    display_step = 1
    with tf.Graph().as_default():
        x = tf.placeholder("float", [None, 784])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        output = inference(x)
        cost = loss(output, y)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = training(cost, learning_rate, global_step)
        eval_op = evaluate(output, y)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('/tmp/tf/logistic_logs', graph_def=sess.graph_def)
            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            # train cycle
            for epoch in range(nrof_training_epoch):
                avg_cost = 0
                total_batch = int(mnist.train.num_examples / batch_size)

                # loop over all batches
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x: mbatch_x, y: mbatch_y}
                    sess.run(train_op, feed_dict=feed_dict)
                    minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                    avg_cost += minibatch_cost / total_batch

                if epoch % display_step == 0:
                    accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                    print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.6f}, validation_accuracy={:.6f}".format(avg_cost, accuracy))

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    saver.save(sess, '/tmp/tf/logistic_logs/model-checkpoint', global_step=global_step)
            print("optimization finished.")

        test_feed_dict = {
            x: mnist.test.images,
            y: mnist.test.labels
        }
        accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
        print("test accuracy:{}".format(accuracy))
if __name__ == '__main__':
    main(sys.argv[1:])
