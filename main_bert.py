import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

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
<<<<<<< HEAD
import pickle

import bert

from bert import run_classifier
from bert import tokenization
=======
from bert import run_classifier
from bert import tokenization
from bert import modeling
import collections
>>>>>>> 408f6370794b1047eead0d570f2469f5ed34cf6b

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

<<<<<<< HEAD


=======
import pickle
>>>>>>> 408f6370794b1047eead0d570f2469f5ed34cf6b
with open('../mapping.pickle', 'rb') as handle:
    mapping = pickle.load(handle)


train = pickle.load(open("train_english.pickle", "rb"))

test = pickle.load(open("test_english.pickle", "rb"))
with open('test_english.pickle', 'rb') as handle:
    test = pickle.load(handle)


class InputFeatures(object):
  def __init__(self, filename, index, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.filename = filename
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
        is_training=True,
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

seq_length = 502
max_seq_length = 502    
sentences_index = {}
features = []
unique_id_to_feature ={}
unique_id = 0
file_name_to_index ={}

# train = test 
for i in range(len(train)):
    i
    if i < 2800:
        continue 
    sentences  = train[i]['sentences']
    file_name=  train[i]["doc_key"]
    # file_names += [file_name]
    file_name_to_index[file_name] = i
    start = 0 
    bert_tokenized = []
    # sentences_index[file_name] = []
    # start_end_id[file_name] = {}
    for i in range(len(sentences)):
        bert_tokenized += sentences[i]
        # end = start + len(sentences[i])-1
        # sentences_index[file_name] += [(start,end)]
        # start = start + len(sentences[i])
    start =0 
    end = 500 
    end = min(end, len(bert_tokenized))
    while (end <= len(bert_tokenized)) :
        tokens_a = bert_tokenized[start:end]
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
        features.append(
            InputFeatures(
                filename=file_name ,
                index = [start,end],
                unique_id=unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
        unique_id +=1
        start += 100
        if end == len(bert_tokenized) : 
            end += 100
        else:
            end += 100
            end = min(end, len(bert_tokenized))
    

for feature in features:
        unique_id_to_feature[feature.unique_id] = feature


LAYERS = [-1,-2,-3,-4]
bert_config_file = "../bert_file/bert_config.json"
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


input_fn = input_fn_builder(features=features, seq_length=max_seq_length)


import collections
import numpy as np
len(features)
count = 0 
# with open('data.json', 'w') as fp:
file_names = []
curr =  ""
map = {}
def try_hey():
    for result in estimator.predict(input_fn, yield_single_examples=True):
        count
        count +=1
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        file_name = feature.filename 
        if file_name not in file_names:
            if curr != "" :
                i = file_name_to_index[curr]
                name = 'bert_test_files/%s' %(str(i))
                with open(name, 'wb') as f:
                    pickle.dump(map, f, pickle.HIGHEST_PROTOCOL)
            map ={}
            file_names += [file_name]
        elif curr != file_name and curr != "":
            i = file_name_to_index[curr]
            name = 'bert_test_files/%s' %(str(i))
            with open(name, 'wb') as f:
                pickle.dump(map, f, pickle.HIGHEST_PROTOCOL)
            i = file_name_to_index[file_name]
            name = 'bert_test_files/%s' %(str(i))
            with open(name, 'rb') as handle:
               map = pickle.load(handle)
        start , end  = feature.index
        for (i, token) in enumerate(feature.tokens):
            if token == "[SEP]" or token == "[CLS]":
                continue
            index = start+i -1
            all_layers = []
            for (j, layer_index) in enumerate(LAYERS):
                layer_output = result["layer_output_%d" % j]
                layers = collections.OrderedDict()
                layers["index"] = layer_index
                layers["values"] = np.array([round(float(x), 6) for x in layer_output[i:(i + 1)].flat])
                all_layers.append(layers)
            if index not in map:
                map[index] = [token , all_layers, 1]
            else:         
                map[index][1][0]["values"]  += all_layers[0]["values"] 
                map[index][1][1]["values"]  += all_layers[1]["values"] 
                map[index][1][2]["values"]  += all_layers[2]["values"] 
                map[index][1][3]["values"]  += all_layers[3]["values"] 
                map[index][2] += 1
        curr = file_name


i = file_name_to_index[curr]
name = 'bert_test_files/%s' %(str(i))
with open(name, 'wb') as f:
            pickle.dump(map, f, pickle.HIGHEST_PROTOCOL)



file_name_to_index = {}
for i in range(len(train)):
    i
    file_name=  train[i]["doc_key"]
    file_name_to_index[file_name] = i


with h5py.File("bert_cache_train.hdf5", "w") as out_file:
    for i in range(len(train)):
        i
        name = 'bert_train_files/%s' %(str(i))
        with open(name, 'rb') as handle:
               map = pickle.load(handle)
        file_name = train[i]["doc_key"]
        group = out_file.create_group(file_name)
        for index in map:
            temp = map[index]
            bert_embd = np.stack([ np.array(temp[1][0]["values"]) /  temp[2]  , np.array(temp[1][1]["values"]) /  temp[2] ,
                np.array(temp[1][2]["values"]) /  temp[2] , np.array(temp[1][3]["values"]) / temp[2] ],1)
            group[str(index)] = bert_embd




with h5py.File("bert_cache_test.hdf5", "w") as out_file:
    for i in range(len(test)):
        i
        name = 'bert_test_files/%s' %(str(i))
        with open(name, 'rb') as handle:
               map = pickle.load(handle)
        file_name = test[i]["doc_key"]
        group = out_file.create_group(file_name)
        for index in map:
            temp = map[index]
            bert_embd = np.stack([ np.array(temp[1][0]["values"]) /  temp[2]  , np.array(temp[1][1]["values"]) /  temp[2] ,
                np.array(temp[1][2]["values"]) /  temp[2] , np.array(temp[1][3]["values"]) / temp[2] ],1)
            bert_embd.shape 
            group[str(index)] = bert_embd

            


with h5py.File("bert_cache.hdf5", "w") as out_file:
    for i in range(len(train)):
        sentences = train[i]["sentences"]
        file_name = train[i]["doc_key"]
        max_sentence_length = max(len(s) for s in sentences)
        context_word_emb = np.zeros([len(sentences), max_sentence_length, 1024 , 4])    
        embedding =  compiled[i]["embedding"]
        group = out_file.create_group(file_name)
        for i in range(len(sentences)):
            if i ==0 :
                temp = 0
            else:
                temp += len(sentences[i-1])
            for j in range(len(sentences[i])):
                index  = temp + j
                context_word_emb[i][j] = np.array(embedding[index])
        group["embedding"] =  context_word_emb



for k in range(len(train)):
        k
        sentences = train[k]["sentences"]
        file_name = train[k]["doc_key"]
        embedding = file[file_name]
        for i in range(len(sentences)):
            if i ==0 :
                temp = 0
            else:
                temp += len(sentences[i-1])
            for j in range(len(sentences[i])):
                index  = temp + j
                if embedding[str(index)][...].shape != (1024,4):
                    break 



file = h5py.File("bert_cache_train.hdf5", "r")










import pickle
with open('files/mapping.pickle', 'rb') as handle:
    mapping = pickle.load(handle)


with open('../dataset/train.english.jsonlines') as f:
    train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

with open('files/train_english.pickle', 'rb') as handle:
    train = pickle.load(handle)



with open('../dataset/dev.english.jsonlines') as f:
    test_examples = [json.loads(jsonline) for jsonline in f.readlines()]


with open('files/test_english.pickle', 'rb') as handle:
    test = pickle.load(handle)


test_examples = train_examples
test = train 
inv_mapping ={}
for i  in range(len(test_examples)):
    t = {}
    file_name =  test_examples[i]["doc_key"]
    s1 = test_examples[i]["sentences"]
    s2 = test[i]["sentences"]
    tokenized = []
    bert_tokenized  = []
    for s  in s1:
        tokenized += s
    for s  in s2:
        bert_tokenized += s
    j = 0 
    for i in range (len(tokenized)):
        # tokenized[i]
        count = mapping[tokenized[i]][1] 
        # count
        for k in range(count):
            t[j] = i 
            j = j + 1 
    inv_mapping[file_name] = t



with open('train_inv_mapping.pickle', 'wb') as handle:
    pickle.dump(inv_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('files/test_inv_mapping.pickle', 'wb') as handle:
    pickle.dump(inv_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
for key in inv_mapping[file_name]:
    bert_tokenized[key], tokenized[inv_mapping[file_name][key]]


