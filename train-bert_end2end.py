import os
import time
import tensorflow as tf
import bert_model as cm
name = "end_to_end"
import util



import run_classifier
import tokenization
import tensorflow_hub as hub

def create_tokenizer_from_hub_module(bert_hub_module_handle,   sess):
    bert_module = hub.Module(bert_hub_module_handle)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)



BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

session = tf.Session()
tokenizer = create_tokenizer_from_hub_module( BERT_MODEL_HUB  , session)



config = util.initialize_from_env("experiments4.conf")
report_frequency = 100
eval_frequency = 5000

model = cm.CorefModel(config, tokenizer, True)

saver = tf.train.Saver()

log_dir = 'logs/' + name
writer = tf.summary.FileWriter(log_dir, flush_secs=20)

max_f1 = 0


session.run(tf.global_variables_initializer())
# model.start_enqueue_thread(session)

accumulated_loss = 0.0
ckpt = tf.train.get_checkpoint_state(log_dir)

if ckpt and ckpt.model_checkpoint_path:
  print("Restoring from: {}".format(ckpt.model_checkpoint_path))
  saver.restore(session, ckpt.model_checkpoint_path)


initial_time = time.time()

import pickle
with open(config["train_path"], 'rb') as handle:
        train_examples = pickle.load(handle)


import  random    
while True :
  try:  
      counter = 0 
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = model.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(model.input_tensors, tensorized_example))
          tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step1, model.train_op] , feed_dict = feed_dict)
          print(str(tf_global_step)+'\r',end='')
          # print(str(tf_global_step))
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
      




