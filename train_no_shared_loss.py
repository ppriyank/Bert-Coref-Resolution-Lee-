import os
import time

import tensorflow as tf
import coref_with_multitask as cm
import util
import random 
from tensorflow.python import debug as tf_debug

config = util.initialize_from_env("experiments.conf")

model = cm.CorefModel(config)
saver = tf.train.Saver()
max_f1 = 0

session = tf.Session() 

session.run(tf.global_variables_initializer())
model.start_enqueue_thread(session)

accumulated_loss = 0.0
accumulated_multitask_loss = 0.0

initial_time = time.time()

if model.is_multitask:
  tf_multitask_loss, tf_global_step, _  = session.run([model.multitask_loss1, model.global_step1, model.multitask_train_op1])
  accumulated_multitask_loss += tf_multitask_loss
else:
  tf_loss, tf_global_step, _  = session.run([model.multitask_loss2, model.global_step2, model.multitask_train_op2])
  accumulated_loss += tf_loss
