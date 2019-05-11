#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import shared_loss as cm
import util
import random 
import logging as log
from tensorflow.python import debug as tf_debug

if __name__ == "__main__":
  config = util.initialize_from_env("experiments.conf")

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  model = cm.CorefModel(config)
  saver = tf.train.Saver()

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  max_swag_acc = 0 
  with tf.Session() as session:
    #session = tf_debug.LocalCLIDebugWrapperSession(session)
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)

    accumulated_loss = 0.0
    accumulated_multitask_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      log.info("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    initial_time = time.time()
    print("We're reporting with frequency: %d" % report_frequency)
    print("We're reporting with eval frequency: %d" % eval_frequency)
    while True:
        tf_loss, tf_global_step, _  = session.run([model.loss, model.global_step1, model.train_op])
        accumulated_loss += tf_loss

        if tf_global_step % report_frequency == 0:
          total_time = time.time() - initial_time
          steps_per_second = tf_global_step / total_time

          average_loss = accumulated_loss / report_frequency
          print("Coreference [{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss[0], steps_per_second))
          writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
          accumulated_loss = 0.0
    
        if tf_global_step % eval_frequency  == 0:
          #saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
          eval_summary, eval_f1, swag_accuracy = model.evaluate(session)
          if eval_f1 > max_f1:
            saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
            max_f1 = eval_f1
            util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
          if swag_accuracy> max_swag_acc:
            saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
            max_swag_acc = swag_accuracy
            util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

          writer.add_summary(eval_summary, tf_global_step)
          writer.add_summary(util.make_summary({"max_eval_f1": max_f1, "max swag accuracy": swag_accuracy}), tf_global_step)

          print("[{}] evaL_f1={:.2f}, max_f1={:.2f}, swag_acc = {:.2f}".format(tf_global_step, eval_f1, max_f1, swag_accuracy))

      
