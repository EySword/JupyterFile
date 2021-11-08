import time
import numpy as np
import tensorflow as tf
import task_forward
import task_backward
import math 
TEST_INTERVAL_SECS = 2

BATCH_SIZE = 50
import numpy as np

def GetInputData():
    x=np.loadtxt('test_forces_sf.dat')
    _y=np.loadtxt('test_forces.dat')
    n=len(_y)
    y=_y.reshape((n,1))
    return x,y

def GetRandomBatch(x,y,count):
    assert x.shape[0] == y.shape[0],('x.shape: %s y.shape: %s' % (x.shape, y.shape))
    if count > x.shape[0]:
        count=x.shape[0]
    _x = x
    _y = y
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    _x = x[perm]
    _y = y[perm]
    return _x[0:count], _y[0:count]


with tf.Graph().as_default() as g:
    x =tf.placeholder(tf.float32,[None,task_forward.INPUT_NODE])
    y_=tf.placeholder(tf.float32,[None,task_forward.OUTPUT_NODE])
    y =task_forward.forward(x,None)

    ema=tf.train.ExponentialMovingAverage(task_backward.MOVING_AVERAGE_DECAY)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)

    accuracy = tf.reduce_mean(tf.square(y - y_))


    while True:
        gpu_options=tf.GPUOptions(allow_growth = True) 
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(task_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #inputX, inputY = GetInputData()
                #xs, ys = GetRandomBatch(inputX,inputY,BATCH_SIZE)
                xs, ys = GetInputData()
                accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_:ys})
                print("After %s training steps,  variance is %g" % (global_step, np.sqrt(accuracy_score)))
                p1=ys
                p2=sess.run(y,feed_dict={x:xs, y_:ys}) 
                np.savetxt('1.dat',p1)
                np.savetxt('2.dat',p2)
            else:
                print("No checkpoint file found.")
        time.sleep(TEST_INTERVAL_SECS)
        break

f1=open('1.dat','r')
f2=open('2.dat','r')
sum1=0
sum2=0
max=0
rmse=0
av=0
natom=3000
for i in range(natom):
  a1=float(f1.readline())
  sum1=sum1+a1 
  a2=float(f2.readline())
  sum2=sum2+a2
  if (max<abs(a1-a2)): max=abs(a1-a2) 
  rmse=rmse+(a1-a2)*(a1-a2)
  av=av+abs(a1-a2)
print(av/natom)
