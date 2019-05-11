import os
import time

import tensorflow as tf

import coref_model as cm
name = "best"
# import coref_model_char as cm
# name = "without_char"
# import coref_model_span_embd as cm
# name = "without_span_embd"
# import coref_model_genre as cm
# name = "without_span_genre"
# import coref_model_speaker as cm
# name = "without_speaker"
# import coref_model_distance_pruning as cm
# name="distance_pruning"

import util





config = util.initialize_from_env()

report_frequency = config["report_frequency"]
eval_frequency = 5000

model = cm.CorefModel(config)
saver = tf.train.Saver()

log_dir = 'logs/' + name
writer = tf.summary.FileWriter(log_dir, flush_secs=20)

max_f1 = 0

session = tf.Session()
session.run(tf.global_variables_initializer())
model.start_enqueue_thread(session)
accumulated_loss = 0.0
ckpt = tf.train.get_checkpoint_state(log_dir)
if ckpt and ckpt.model_checkpoint_path:
  print("Restoring from: {}".format(ckpt.model_checkpoint_path))
  saver.restore(session, ckpt.model_checkpoint_path)


initial_time = time.time()

try:  
    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss
      if tf_global_step % report_frequency == 0:
          total_time = time.time() - initial_time
          steps_per_second = tf_global_step / total_time
          average_loss = accumulated_loss / report_frequency
          print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
          writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
          accumulated_loss = 0.0
      if tf_global_step % eval_frequency  == 0:
        #saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session)
        if eval_f1 > max_f1:
          saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
          print("====")
except Exception as e: 
    print(e)



