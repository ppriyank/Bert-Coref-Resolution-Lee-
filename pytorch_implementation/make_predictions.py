"""
This is the BERT part of the ELMo. 

"""
import torch 
import allennlp
from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader
from conll_coref_reader_bert import ConllCorefBertReader
from swag_reader import SWAGDatasetReader
#present in allennlp 0.8.4
from multitask_sampling_trainer import  MultiTaskTrainer
#import allennlp.data.token_indexers.wordpiece_indexer.PretrainedBertIndexer as BertIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp_coref import CoreferenceResolver
from swag_model import SWAGExampleModel
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.trainer import Trainer
from allennlp.modules import FeedForward
from elmo_text_field_embedder import ElmoTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from torch import optim
from pathlib import Path
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder
from pytorch_pretrained_bert.modeling import BertModel
from typing import Dict
import torch.nn as nn
import pickle
import copy
from pytorch_pretrained_bert.optimization import BertAdam

import logging as log

#directory = "/beegfs/yp913/Bert-Coref-Resolution-Lee-/"
directory = "/scratch/ovd208/nlu/bert/Bert-Coref-Resolution-Lee-/"

#dataset_folder = "/beegfs/yp913/dataset/"
dataset_folder = "/scratch/ovd208/nlu/coref_lee_data/"

class PretrainedBertModel:
    """
    In some instances you may want to load the same BERT model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """
    _cache: Dict[str, BertModel] = {}

    @classmethod
    def load(cls, model_name: str, cache_model: bool = True) -> BertModel:
        if model_name in cls._cache:
            return PretrainedBertModel._cache[model_name]

        model = BertModel.from_pretrained(model_name)
        if cache_model:
            cls._cache[model_name] = model

        return model

class PretrainedBertEmbedder(BertEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .tar.gz file with the model weights.
        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L41
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of BERT parameters for fine tuning.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    """
    def __init__(self, pretrained_model: str, requires_grad: bool = False, top_layer_only: bool = False) -> None:
        model = PretrainedBertModel.load(pretrained_model)
        # Only train last 4 layers
        for name, param in model.named_parameters():
            if "encoder.layer.11" in name or "encoder.layer.10" in name or "encoder.layer.9" in name or "encoder.layer.8" in name:
                print("making finetuning true ")
                param.requires_grad = requires_grad
            else:
                param.requires_grad = False
        super().__init__(bert_model=model, top_layer_only=top_layer_only)


def train_only_lee():
    # This is WORKING! 
	# load datasetreader 
    # Save logging to a local file
    # Multitasking
    log.getLogger().addHandler(log.FileHandler(directory+"/log.log"))

    lr = 0.00001
    batch_size = 2
    epochs = 100
    max_seq_len = 512
    max_span_width = 30
    #token_indexer = BertIndexer(pretrained_model="bert-base-uncased", max_pieces=max_seq_len, do_lowercase=True,)
    token_indexer = PretrainedBertIndexer("bert-base-cased", do_lowercase=False)
    reader = ConllCorefBertReader(max_span_width = max_span_width, token_indexers = {"tokens": token_indexer})

    EMBEDDING_DIM = 1024
    HIDDEN_DIM = 200
    processed_reader_dir = Path(directory+"processed/")
    
    train_ds = None
    if processed_reader_dir.is_dir():
        print("Loading indexed from checkpoints")
        train_path =  Path(directory +"processed/train_d")
        if train_path.exists():
            train_ds = pickle.load(open(directory + "processed/conll/train_d", "rb"))
            val_ds =  pickle.load(open(directory + "processed/conll/val_d", "rb"))
            test_ds = pickle.load(open(directory + "processed/conll/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (reader.read(dataset_folder + fname) for fname in ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
            pickle.dump(train_ds,open(directory + "processed/train_d", "wb"))
            pickle.dump(val_ds,open(directory + "processed/val_d", "wb"))
            pickle.dump(test_ds,open(directory + "processed/test_d", "wb"))
            print("saved checkpoints")
    # restore checkpoint here

    #vocab = Vocabulary.from_instances(train_ds + val_ds)
    vocab = Vocabulary()
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    val_iterator = BasicIterator(batch_size=batch_size)
    val_iterator.index_with(vocab)
    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

    bert_embedder = PretrainedBertEmbedder(
             pretrained_model="bert-base-cased",
             top_layer_only=True, # conserve memory
             requires_grad=True
     )
    # here, allow_unmatched_key = True since we dont pass in offsets since 
    #we allow for word embedings of the bert-tokenized, wnot necessiarly the 
    # original tokens
    # see the documetnation for offsets here for more info:
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/bert_token_embedder.py
    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys=True)
    BERT_DIM = word_embedding.get_output_dim()
    # at each batch, sample from the two, and load th eLSTM
    shared_layer = torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
    seq2seq = PytorchSeq2SeqWrapper(shared_layer)
    mention_feedforward = FeedForward(input_dim = 2336, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    antecedent_feedforward = FeedForward(input_dim = 7776, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())

    model = CoreferenceResolver(vocab=vocab, text_field_embedder=word_embedding,context_layer= seq2seq, mention_feedforward=mention_feedforward,antecedent_feedforward=antecedent_feedforward , feature_size=768,max_span_width=max_span_width,spans_per_word=0.4,max_antecedents=250,lexical_dropout= 0.2)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # and then we can do the shared loss
    # 
    # Get 
    USE_GPU = 0
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator = val_iterator, 
        train_dataset=train_ds,
        validation_dataset = val_ds, 
        validation_metric = "+coref_f1",
        cuda_device=0 if USE_GPU else -1,
        serialization_dir= directory + "saved_models/only_lee",
        num_epochs=epochs,
    )    

    metrics = trainer.train()
    # save the model
    with open(directory + "saved_models/current_run_model_state", 'wb') as f:
        torch.save(model.state_dict(), f)

def load_swag(swag_reader, path):
    swag_reader_dir =  Path(path+"processed/swag/")
    directory = path
    if swag_reader_dir.is_dir():
        print("Loading indexed from checkpoints for Ontonotes")
        train_path =  Path(directory +"processed/swag/train_d")
        if train_path.exists():
            train_ds = pickle.load(open(directory + "processed/swag/train_d", "rb"))
            val_ds =  pickle.load(open(directory + "processed/swag/val_d", "rb"))
            test_ds = pickle.load(open(directory + "processed/swag/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (swag_reader._read(directory + "data/swagaf/" + fname) for fname in ["train.csv", "val.csv", "test.csv"])
            train_ds = list(train_ds)
            val_ds = list(val_ds)
            test_ds = list(test_ds)
            pickle.dump(train_ds,open(directory + "processed/swag/train_d", "wb"))
            pickle.dump(val_ds,open(directory + "processed/swag/val_d", "wb"))
            pickle.dump(test_ds,open(directory + "processed/swag/test_d", "wb"))
            print("saved checkpoints")
        swag_datasets = [train_ds, val_ds, test_ds]
        return swag_datasets
def load_datasets(conll_reader, swag_reader, path):
    swag_datasets = load_swag(swag_reader, path)
    conll_reader_dir =  Path(path+"processed/conll/")
    if conll_reader_dir.is_dir():
        print("Loading indexed from checkpoints for Ontonotes")
        train_path =  Path(directory +"processed/conll/train_d")
        if train_path.exists():
            train_ds = pickle.load(open(directory + "processed/conll/train_d", "rb"))
            val_ds =  pickle.load(open(directory + "processed/conll/val_d", "rb"))
            test_ds = pickle.load(open(directory + "processed/conll/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (conll_reader.read(dataset_folder + fname) for fname in ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
            pickle.dump(train_ds,open(directory + "processed/conll/train_d", "wb"))
            pickle.dump(val_ds,open(directory + "processed/conll/val_d", "wb"))
            pickle.dump(test_ds,open(directory + "processed/conll/test_d", "wb"))
            print("saved checkpoints")
        conll_datasets = [train_ds, val_ds, test_ds]
    return conll_datasets, swag_datasets

def train_only_swag():
    # load datasetreader 
    # Save logging to a local file
    # Multitasking
    log.getLogger().addHandler(log.FileHandler(directory+"/log.log"))

    lr = 0.00001
    batch_size = 2
    epochs = 100
    max_seq_len = 512
    max_span_width = 30
    #token_indexer = BertIndexer(pretrained_model="bert-base-uncased", max_pieces=max_seq_len, do_lowercase=True,)
    token_indexer = PretrainedBertIndexer("bert-base-cased", do_lowercase=False)
    swag_reader = SWAGDatasetReader(tokenizer=token_indexer.wordpiece_tokenizer,lazy=True, token_indexers=token_indexer)
    EMBEDDING_DIM = 1024
    HIDDEN_DIM = 200
    swag_datasets = load_swag(swag_reader, directory)
    swag_vocab = Vocabulary()

    swag_vocab = Vocabulary()
    swag_iterator = BasicIterator(batch_size=batch_size)
    swag_iterator.index_with(swag_vocab)

    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

    bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-cased",top_layer_only=True, requires_grad=True)

    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys=True)
    BERT_DIM = word_embedding.get_output_dim()
    seq2vec = PytorchSeq2VecWrapper(torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))
    mention_feedforward = FeedForward(input_dim = 2336, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    antecedent_feedforward = FeedForward(input_dim = 7776, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())

    model = SWAGExampleModel(vocab=swag_vocab, text_field_embedder=word_embedding, phrase_encoder=seq2vec)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    USE_GPU =1 
    val_iterator = swag_iterator(swag_datasets[1], num_epochs=1, shuffle=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=swag_iterator,
        validation_iterator = swag_iterator, 
        train_dataset=swag_datasets[0],
        validation_dataset = swag_datasets[1], 
        validation_metric = "+accuracy",
        cuda_device=0 if USE_GPU else -1,
        serialization_dir= directory + "saved_models/current_run_model_state_swag",
        num_epochs=epochs,
    )    

    metrics = trainer.train()
    # save the model
    with open(directory + "saved_models/current_run_model_state", 'wb') as f:
        torch.save(model.state_dict(), f)


def predict():
    # load datasetreader 
    # Save logging to a local file
    # Multitasking
    log.getLogger().addHandler(log.FileHandler(directory+"/log.log"))

    batch_size = 1
    epochs = 10 
    max_seq_len = 512
    max_span_width = 30
    #token_indexer = BertIndexer(pretrained_model="bert-base-uncased", max_pieces=max_seq_len, do_lowercase=True,)
    token_indexer = PretrainedBertIndexer("bert-base-cased", do_lowercase=False)
    conll_reader = ConllCorefBertReader(max_span_width = max_span_width, token_indexers = {"tokens": token_indexer})
    EMBEDDING_DIM = 1024
    HIDDEN_DIM = 200

    path = directory

    conll_reader_dir =  Path(path+"processed/conll/")
    conll_datasets = None
    if conll_reader_dir.is_dir():
        print("Loading indexed from checkpoints for Ontonotes")
        train_path =  Path(directory +"processed/conll/train_d")
        if train_path.exists():
            train_ds = pickle.load(open(directory + "processed/conll/train_d", "rb"))
            val_ds =  pickle.load(open(directory + "processed/conll/val_d", "rb"))
            test_ds = pickle.load(open(directory + "processed/conll/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (conll_reader.read(dataset_folder + fname) for fname in ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
            pickle.dump(train_ds,open(directory + "processed/conll/train_d", "wb"))
            pickle.dump(val_ds,open(directory + "processed/conll/val_d", "wb"))
            pickle.dump(test_ds,open(directory + "processed/conll/test_d", "wb"))
            print("saved checkpoints")
        conll_datasets = [train_ds, val_ds, test_ds]
    
    conll_vocab = Vocabulary()
    conll_iterator = BasicIterator(batch_size=batch_size)
    conll_iterator.index_with(conll_vocab)


    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

    bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-cased",top_layer_only=True, requires_grad=True)

    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys=True)
    BERT_DIM = word_embedding.get_output_dim()

    seq2seq = PytorchSeq2SeqWrapper(torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))
    seq2vec = PytorchSeq2VecWrapper(torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))
    mention_feedforward = FeedForward(input_dim = 2336, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    antecedent_feedforward = FeedForward(input_dim = 7776, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    model1 = CoreferenceResolver(vocab=conll_vocab, text_field_embedder=word_embedding,context_layer= seq2seq, mention_feedforward=mention_feedforward,antecedent_feedforward=antecedent_feedforward , feature_size=768,max_span_width=max_span_width,spans_per_word=0.4,max_antecedents=250,lexical_dropout= 0.2)

    conll_test_iterator = conll_iterator(conll_datasets[2], num_epochs=1, shuffle=False)
    USE_GPU = 1

    serialization_dir=directory + "saved_models/multitask/"

    TRAINED_MODEL_PATH = directory + "saved_models/multitask/conll/model_state_epoch_9.th"

    #TRAINED_MODEL_PATH = directory + "saved_models/current_run_model_state/model_state_epoch_99.th"

    model1.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    model1.eval()

    num_batches = len(conll_datasets[2])

    for i in range(20):
        batch = next(conll_test_iterator, None)
        output = model1.forward(**batch)
        #import pdb
        #pdb.set_trace()
        if len(batch['metadata'][0]['original_text']) > 150:
            continue
        #let us print out the predictions in the first document of this batch
        pairs = []
        for index,j in enumerate(output['predicted_antecedents'][0]):
            if j is not -1:
                i1 = output['top_spans'][0][index]
                i2 = output['top_spans'][0][output['antecedent_indices'][index][j]]
                d0 = output['document'][0]
                pairs.append([d0[i1[0]:i1[1]+1], d0[i2[0]:i2[1]+1]])
        #pairs
        print(batch['metadata'][0]['original_text'])
        print( batch['metadata'][0]['clusters'])
        print(pairs)
        metrics = model1.get_metrics()
        print(metrics['coref_f1'])
        #import pdb
        #pdb.set_trace()


def predict_only_lee():
    # load datasetreader 
    # Save logging to a local file
    # Multitasking
    log.getLogger().addHandler(log.FileHandler(directory+"/log.log"))

    batch_size = 1
    epochs = 10 
    max_seq_len = 512
    max_span_width = 30
    #token_indexer = BertIndexer(pretrained_model="bert-base-uncased", max_pieces=max_seq_len, do_lowercase=True,)
    token_indexer = PretrainedBertIndexer("bert-base-cased", do_lowercase=False)
    conll_reader = ConllCorefBertReader(max_span_width = max_span_width, token_indexers = {"tokens": token_indexer})
    EMBEDDING_DIM = 1024
    HIDDEN_DIM = 200
    processed_reader_dir = Path(directory+"processed/")

    train_ds = None
    test_ds=None
    if processed_reader_dir.is_dir():
        print("Loading indexed from checkpoints")
        train_path =  Path(directory +"processed/train_d")
        if train_path.exists():
            train_ds = pickle.load(open(directory + "processed/conll/train_d", "rb"))
            val_ds =  pickle.load(open(directory + "processed/conll/val_d", "rb"))
            test_ds = pickle.load(open(directory + "processed/conll/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (reader.read(dataset_folder + fname) for fname in ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
            pickle.dump(train_ds,open(directory + "processed/train_d", "wb"))
            pickle.dump(val_ds,open(directory + "processed/val_d", "wb"))
            pickle.dump(test_ds,open(directory + "processed/test_d", "wb"))
            print("saved checkpoints")

    
    vocab = Vocabulary()
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

    bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-cased",top_layer_only=True, requires_grad=True)

    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys=True)
    BERT_DIM = word_embedding.get_output_dim()

    shared_layer = torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)

    seq2seq = PytorchSeq2SeqWrapper(shared_layer)
    #seq2vec = PytorchSeq2VecWrapper(torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))
    mention_feedforward = FeedForward(input_dim = 2336, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    antecedent_feedforward = FeedForward(input_dim = 7776, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    model1 = CoreferenceResolver(vocab=vocab, text_field_embedder=word_embedding,context_layer= seq2seq, mention_feedforward=mention_feedforward,antecedent_feedforward=antecedent_feedforward , feature_size=768,max_span_width=max_span_width,spans_per_word=0.4,max_antecedents=250,lexical_dropout= 0.2)

    conll_test_iterator = iterator(test_ds, num_epochs=1, shuffle=False)
    USE_GPU = 1

    #serialization_dir=directory + "saved_models/multitask/"

    #TRAINED_MODEL_PATH = directory + "saved_models/multitask/conll/model_state_epoch_9.th"

    TRAINED_MODEL_PATH = directory + "saved_models/current_run_model_state/model_state_epoch_99.th"

    model1.eval()
    model1.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    model1.eval()

    num_batches = len(test_ds)

    for i in range(20):
        batch = next(conll_test_iterator, None)
        output = model1.forward(**batch)
        
        #let us print out the predictions in the first document of this batch
        pairs = []
        for index,j in enumerate(output['predicted_antecedents'][0]):
            if j is not -1:
                i1 = output['top_spans'][0][index]
                i2 = output['top_spans'][0][output['antecedent_indices'][index][j]]
                d0 = output['document'][0]
                pairs.append([d0[i1[0]:i1[1]+1], d0[i2[0]:i2[1]+1]])

        #pairs
        #print(pairs)
        metrics = model1.get_metrics()
        print(metrics['coref_f1'])
        #import pdb
        #pdb.set_trace()

#predict_only_lee()
predict()
