import tensorflow as tf
import task_forward
import os
import data_input

MODEL_SAVE_PATH='./model1/'
MODEL_NAME='task_model'

BATCH_SIZE =50 
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY =0.99
REGULARIZER = 0.0001
STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

def split():
    f1 = open('forces_sf.dat')
    f2 = open('forces.dat')

    f3 = open('train_forces_sf.dat', 'x')
    f4 = open('train_forces.dat', 'x')

    f5 = open('test_forces_sf.dat', 'x')
    f6 = open('test_forces.dat', 'x')

    for i in range(500):
        f3.write(f1.readline())
        f4.write(f2.readline())
    for i in range(40000):
        f1.readline()
        f2.readline()
    for i in range(10000):
        f5.write(f1.readline())
        f6.write(f2.readline())

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()

def backward():
    x=tf.placeholder(tf.float32,[None,task_forward.INPUT_NODE], name = "x")
    y_=tf.placeholder(tf.float32,[None,task_forward.OUTPUT_NODE], name = "y")
    y=task_forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable = False)

    mse=tf.reduce_mean(tf.square(y_ - y))
    loss=mse + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,5000/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)

    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name = 'train')


    saver=tf.train.Saver()
    
    #gpu_options=tf.GPUOptions(allow_growth = True) 
    #config = tf.ConfigProto(gpu_options=gpu_options)
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        inputX, inputY = data_input.GetInputData()
        print(inputX.shape)
        print(inputY.shape)
        
        for i in range(STEPS):
            xs, ys = data_input.GetRandomBatch(inputX,inputY,BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x:xs,y_:ys})
            if i % 100 == 0:
                print("After %d training steps, loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    split()
    backward()

if __name__ == '__main__':
    main()
