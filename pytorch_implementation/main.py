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
import pickle

def main():
	# load datasetreader 

    lr = 0.001
    batch_size = 4
    epochs = 100    
    max_seq_len = 512
    max_span_width = 30
    #token_indexer = BertIndexer(pretrained_model="bert-base-uncased", max_pieces=max_seq_len, do_lowercase=True,)
    token_indexer = PretrainedBertIndexer("bert-large-uncased", do_lowercase=True)
    reader = ConllCorefBertReader(max_span_width = max_span_width, token_indexers = {"tokens": token_indexer})

    EMBEDDING_DIM = 1024
    HIDDEN_DIM = 200
    processed_reader_dir = Path("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/")
    if processed_reader_dir.is_dir():
        print("Loading indexed from checkpoints")
        train_path =  Path("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/train_d")
        if train_path.exists():
            train_ds = pickle.load(open("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/train_d", "rb"))
            val_ds =  pickle.load(open("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/val_d", "rb"))
            test_ds = pickle.load(open("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/test_d", "rb"))
        else:
            print("checkpoints not found")
            train_ds, val_ds, test_ds = (reader.read("/beegfs/yp913/dataset/" + fname) for fname in ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
            pickle.dump(train_ds,open("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/train_d", "wb"))
            pickle.dump(val_ds,open("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/val_d", "wb"))
            pickle.dump(test_ds,open("/beegfs/yp913/Bert-Coref-Resolution-Lee-/processed/test_d", "wb"))
            print("saved checkpoints")
    # restore checkpoint here

    #vocab = Vocabulary.from_instances(train_ds + val_ds)
    vocab = Vocabulary()
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
    from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

    bert_embedder = PretrainedBertEmbedder(
             pretrained_model="bert-large-uncased",
             top_layer_only=True, # conserve memory
     )
    # TODO: Make sure that BERT is being fintuned, requires_grad=True.
    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys = True)
    BERT_DIM = word_embedding.get_output_dim()
    #token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),embedding_dim=EMBEDDING_DIM)
    #tokenizer = allennlp.modules.token_embedders.token_embedder.TokenEmbedder()
    #text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding}, embedder_to_indexer_map=,allow_unmatched_keys=True)
    # BiLSTM

    seq2seq = PytorchSeq2SeqWrapper(torch.nn.LSTM(BERT_DIM, HIDDEN_DIM, batch_first=True))
    mention_feedforward = FeedForward(input_dim = 2192, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())
    antecedent_feedforward = FeedForward(input_dim = 7344, num_layers = 2, hidden_dims = 150, activations = torch.nn.ReLU())

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


main()
