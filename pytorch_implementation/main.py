"""
This is the BERT part of the ELMo. 

"""
import torch 
import allennlp
from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader
from conll_coref_reader_bert import ConllCorefBertReader
#present in allennlp 0.8.4
#import allennlp.data.token_indexers.wordpiece_indexer.PretrainedBertIndexer as BertIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp_coref import CoreferenceResolver

from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.trainer import Trainer
from allennlp.modules import FeedForward
from elmo_text_field_embedder import ElmoTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from torch import optim
from pathlib import Path
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder
from pytorch_pretrained_bert.modeling import BertModel
from typing import Dict
import torch.nn as nn
import pickle
from pytorch_pretrained_bert.optimization import BertAdam

import logging as log

directory = "/beegfs/yp913/Bert-Coref-Resolution-Lee-/"
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

def load_model_state(model, state_path, gpu_id, skip_task_models=[], strict=True):
    ''' Helper function to load a model state

    Parameters
    ----------
    model: The model object to populate with loaded parameters.
    state_path: The path to a model_state checkpoint.
    gpu_id: The GPU to use. -1 for no GPU.
    skip_task_models: If set, skip task-specific parameters for these tasks.
        This does not necessarily skip loading ELMo scalar weights, but I (Sam) sincerely
        doubt that this matters.
    strict: Whether we should fail if any parameters aren't found in the checkpoint. If false,
        there is a risk of leaving some parameters in their randomly initialized state.
    '''
    model_state = torch.load(state_path, map_location=device_mapping(gpu_id))

    assert_for_log(
        not (
            skip_task_models and strict),
        "Can't skip task models while also strictly loading task models. Something is wrong.")

    for name, param in model.named_parameters():
        # Make sure no trainable params are missing.
        if param.requires_grad:
            if strict:
                assert_for_log(name in model_state,
                               "In strict mode and failed to find at least one parameter: " + name)
            elif (name not in model_state) and ((not skip_task_models) or ("_mdl" not in name)):
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                logging.error("Parameter missing from checkpoint: " + name)
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if skip_task_models:
        keys_to_skip = []
        for task in skip_task_models:
            new_keys_to_skip = [key for key in model_state if "%s_mdl" % task in key]
            if new_keys_to_skip:
                logging.info("Skipping task-specific parameters for task: %s" % task)
                keys_to_skip += new_keys_to_skip
            else:
                logging.info("Found no task-specific parameters to skip for task: %s" % task)
        for key in keys_to_skip:
            del model_state[key]

    model.load_state_dict(model_state, strict=False)
    logging.info("Loaded model state from %s", state_path)


def main():
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
    token_indexer = PretrainedBertIndexer("bert-base-uncased", do_lowercase=True)
    reader = ConllCorefBertReader(max_span_width = max_span_width, token_indexers = {"tokens": token_indexer})

    EMBEDDING_DIM = 1024
    HIDDEN_DIM = 200
    processed_reader_dir = Path(directory+"processed/")
    if processed_reader_dir.is_dir():
        print("Loading indexed from checkpoints")
        train_path =  Path(directory +"processed/train_d")
        if train_path.exists():
            train_ds = pickle.load(open(directory + "processed/train_d", "rb"))
            val_ds =  pickle.load(open(directory + "processed/val_d", "rb"))
            test_ds = pickle.load(open(directory + "processed/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (reader.read("/beegfs/yp913/dataset/" + fname) for fname in ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
            pickle.dump(train_ds,open(directory + "processed/train_d", "wb"))
            pickle.dump(val_ds,open(directory + "processed/val_d", "wb"))
            pickle.dump(test_ds,open(directory + "processed/test_d", "wb"))
            print("saved checkpoints")
    # restore checkpoint here

    #vocab = Vocabulary.from_instances(train_ds + val_ds)
    vocab = Vocabulary()
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

    bert_embedder = PretrainedBertEmbedder(
             pretrained_model="bert-base-uncased",
             top_layer_only=True, # conserve memory
             requires_grad=True
     )
    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys = True)
    BERT_DIM = word_embedding.get_output_dim()

    seq2seq = PytorchSeq2SeqWrapper(torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True))
    mention_feedforward = FeedForward(input_dim = 1936, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    antecedent_feedforward = FeedForward(input_dim = 6576, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())

    model = CoreferenceResolver(vocab=vocab, text_field_embedder=word_embedding,context_layer= seq2seq, mention_feedforward=mention_feedforward,antecedent_feedforward=antecedent_feedforward , feature_size=768,max_span_width=max_span_width,spans_per_word=0.4,max_antecedents=250,lexical_dropout= 0.2)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    USE_GPU = 1
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        cuda_device=0 if USE_GPU else -1,
        num_epochs=epochs,
    )    

    metrics = trainer.train()



class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(10, 1)
        self.fc2 = nn.Linear(10, 1)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3
    

# Train modelA
modelA = MyModelA()
x = torch.randn(1, 10)
output1, output2, output3 = modelA(x)
# ...

# Save modelA
torch.save(modelA.state_dict(), 'modelA.pth')


# Duplicate modelA and add a switch for inference
class MyModelB(nn.Module):
    def __init__(self, fast_inference=False):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(10, 1)
        self.fc2 = nn.Linear(10, 1)
        self.fc3 = nn.Linear(10, 1)
        self.fast_inference = fast_inference
        
    def forward(self, x):
        if self.fast_inference:
            return self.fc1(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3

modelB = MyModelB(fast_inference=True)
modelB.load_state_dict(torch.load('modelA.pth'))
# load_state_dict adn then you can just give htis, and hten you switch it. 
output = modelB(x)


main()
