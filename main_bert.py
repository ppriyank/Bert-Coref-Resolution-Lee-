
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bert
import pickle
import h5py
import json
import sys
sys.path.append("../..")
import os
from bert import run_classifier
from bert import tokenization
from bert import modeling
import collections

def create_tokenizer_from_hub_module(bert_hub_module_handle,   sess):
    bert_module = hub.Module(bert_hub_module_handle)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
    return tokenization.FullTokenizer(
    	  vocab_file=vocab_file, do_lower_case=do_lower_case)



BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


args = ["train.english.jsonlines" , "dev.english.jsonlines"]
json_filename = args[1]
data_path = json_filename

session = tf.Session()
tokenizer = create_tokenizer_from_hub_module( BERT_MODEL_HUB  , session)

tokens = []
unique_tokens = set([])

for data_path in args : 
	with open(data_path) as in_file:
	  for doc_num, line in enumerate(in_file.readlines()):
	    example = json.loads(line)
	    sentences = example["sentences"]
	    for sentence in sentences :
	    	tokens += sentence
	    unique_tokens = unique_tokens.union(set(tokens))

	    
mapping = {}
# inv_mapping = {}
for x in unique_tokens : 
	temp = tokenizer.tokenize(x) 
	mapping[x] = [temp , len(temp)]
	# inv_mapping[temp] = [x , len(temp)] 


with open('mapping.pickle', 'wb') as handle:
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


train = []
data_path = args[1]
import  util 
import json
with open(data_path) as in_file:
  for doc_num, line in enumerate(in_file.readlines()):
    example = json.loads(line)
    sentences = example["sentences"]
    tokens = []
    file_mapping = {}
    bert_tokens = []
    bert_sent =  [" ".join(sent) for sent in sentences]
    bert_tokenized = [tokenizer.tokenize(sent) for sent in bert_sent ]
    for sentence in sentences :
    	tokens += sentence
    # for bert_token in bert_tokenized : 
    # 	bert_tokens += bert_token
    till = 0
    for i in range(len(tokens)):
    	file_mapping[i] = [till]
    	till += mapping[tokens[i]][1]
    	file_mapping[i] += [till-1]
    bert_clusters = []
    clusters = example["clusters"]
    for cluster in clusters:
    	temp = []
    	for span in cluster :
    		start = file_mapping[span[0]][0]
    		end =  file_mapping[span[1]][1]
    		temp += [[start,end]]
    	bert_clusters += [temp]
    example['sentences'] =  bert_tokenized
    example['clusters'] =  bert_clusters
    # json.dump(example, fp)
    train += [example]



with open('test_english.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)



gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}

bert_cluster = train[-1]['clusters']
bert_tokenized = train[-1]['sentences']
bert_tokens =  []
for bert_token in bert_tokenized : 
        bert_tokens += bert_token
    

bert_mentions = sorted(tuple(m) for m in util.flatten(bert_cluster))
bert_gold_mention_map = {m:i for i,m in enumerate(bert_mentions)}


for i in range(len(gold_mentions)):
    span = gold_mentions[i]
    bert_span = bert_mentions[i]
    tokens[span[0]:span[1]+1]   , bert_tokens[bert_span[0] : bert_span[1]+1]






##################################################################################################################################################################
##################################################################################################################################################################


with open('mapping.pickle', 'rb') as handle:
    mapping = pickle.load(handle)


with open('train_english.pickle', 'rb') as handle:
    train = pickle.load(handle)


with open('test_english.pickle', 'rb') as handle:
    test = pickle.load(handle)


class InputFeatures(object):
  def __init__(self, index, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.index = index
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))
    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:
      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()
      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    all_layers = model.get_all_encoder_layers()
    predictions = {
        "unique_id": unique_ids,
    }
    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec
  return model_fn



def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []
  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    num_examples = len(features)
    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })
    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d
  return input_fn

# creating bert embedding 
seq_length = 502
sentences_index = {}
max_seq_length = 502    
total_length = []
# total_sentence_count  = []
# total_length_sentence  = []
features = {}
file_names = []
for i in range(len(train)):
    unique_id = 0
    sentences  = train[i]['sentences']
    file_name=  train[i]["doc_key"]
    file_names += [file_name]
    start = 0 
    # bert_tokenized = []
    sentences_index[file_name] = []
    for i in range(len(sentences)):
        # total_length_sentence.append(len(sentences[i]))
        bert_tokenized += sentences[i]
        end = start + len(sentences[i])-1
        sentences_index[file_name] += [(start,end)]
        # bert_tokenized[start:end+1] , sentences[i]
        start = start + len(sentences[i])
    # total_length.append(len(bert_tokenized))
    # total_sentence_count.append(len(sentences_index[file_name]))
    start =0 
    end = 500 
    end = min(end, len(bert_tokenized))
    features[file_name] = []
    while (end <= len(bert_tokenized)) :
        tokens_a = bert_tokenized[start:end]
        tokens_a
        start += 100
        if end == len(bert_tokenized) : 
            end += 100
        else:
            end += 100
            end = min(end, len(bert_tokenized))
        # len(tokens_a)
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        features[file_name].append(
            InputFeatures(
                index = [start,end],
                unique_id=unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
        start += 100
        if end == len(bert_tokenized) : 
            end += 100
        else:
            end += 100
            end = min(end, len(bert_tokenized))
        




LAYERS = [-1,-2,-3,-4]
bert_config_file = "bert_file/bert_config.json"
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
num_tpu_cores=8
master= None
use_tpu = False
batch_size = 32  
vocab_file ="bert_file/vocab.txt"
use_one_hot_embeddings = False
init_checkpoint=  "bert_file/bert_model.ckpt"
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
      master=master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=num_tpu_cores,
          per_host_input_for_training=is_per_host))



model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      layer_indexes=LAYERS,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_one_hot_embeddings)



estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=batch_size)


max_seq_length = 502
compiled = []

for k in range(len(train)):
    file_name=  train[k]["doc_key"]
    curr= features[file_name]
    all_features = {}
    for j in curr: 
        start = j.index[0]
        end  = j.index[1]
        # start, end 
        input_fn = input_fn_builder(
            features=[j], seq_length=max_seq_length)
        for result in estimator.predict(input_fn, yield_single_examples=True):
            for (i, token) in enumerate(j.tokens):
                if token == "[SEP]" or token == "[CLS]":
                    continue
                index = start+i -1
                # index , token
                all_layers = []
                for (j, layer_index) in enumerate(LAYERS):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]
                    all_layers.append(layers)
                if index not in map:
                    map[index] = [token , all_layers, 1]
                else:         
                    map[index][1][0]["values"]  += all_layers[0]["values"] 
                    map[index][1][1]["values"]  += all_layers[1]["values"] 
                    map[index][1][2]["values"]  += all_layers[2]["values"] 
                    map[index][1][3]["values"]  += all_layers[3]["values"] 
                    map[index][2] += 1
    for index in map:
        token = map[index][0]
        bert_embd = tf.stack([ np.array(map[index][1][0]["values"]) /  map[index][2] , np.array(map[index][1][1]["values"]) /  map[index][2] ,
         np.array(map[index][1][2]["values"]) /  map[index][2] , np.array(map[index][1][3]["values"]) /  map[index][2] ] , -1)
        all_features[index] = [ token , bert_embd ]
    compiled.append({})
    compiled[k]["sentences"] = train[k]["sentences"]
    compiled[k]['doc_key'] = train[k]['doc_key']
    file_name = train[i]['doc_key']
    compiled[k]['sent_index'] = sentences_index[file_name] 
    compiled[k]['clusters'] = train[k]['clusters']
    compiled[k]['embedding'] = all_features



with open('train_bert.pickle', 'wb') as handle:
    pickle.dump(compiled, handle, protocol=pickle.HIGHEST_PROTOCOL)
