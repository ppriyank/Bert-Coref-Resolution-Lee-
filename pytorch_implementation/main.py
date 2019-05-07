import torch 
import allennlp
from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader
import allennlp.data.token_indexers.wordpiece_indexer.PretrainedBertIndexer as BertIndexer
from allennlp_coref import CoreferenceResolver

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.trainer import Trainer
from allennlp.modules import FeedForward
from elmo_text_field_embedder import ElmoTextFieldEmbedder
def main():
	# load datasetreader 

    lr= 0.001
    batch_size = 512
    epochs = 100    
    max_seq_len = 512

    token_indexer = BertIndexer(pretrained_model="bert-base-uncased",
                                max_pieces=max_seq_len,
                                do_lowercase=True,)
    def tokenizer(s: str):
        return token_indexer.wordpiece_tokenizer(s)[:max_seq_len - 2] 

    reader = ConllCoreReader(tokenizer = tokenizer, token_indexers = {"tokens": token_indexer})


    train_ds, val_ds, test_ds = (reader.read("/beegfs/yp913/Bert-Coref-Resolution-Lee-/" + fname) for fname in 
                                ["train.english.v4_gold_conll", "dev.english.v4_gold_conll", "test.english.v4_gold_conll"])
    vocab = Vocabulary.from_instances(train_ds + val_ds)
    iterator = BucketIterator(batch_size=batch_size, biggest_batch_first=True, sorting_keys=[("tokens", "num_tokens")],)
    iterator.index_with(vocab)
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
    tokenizer = allennlp.modules.token_embedders.token_embedder.TokenEmbedder()
    text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    seq2seq = Seq2SeqEncoder()

    model = CoreferenceResolver(vocab=Vocabulary,
                 text_field_embedder=text_field_embedder,
                 context_layer= seq2seq,
                 mention_feedforward= FeedForward(),
                 antecedent_feedforward= FeedForward(),
                 feature_size= 1024,
                 max_span_width=10,
                 spans_per_word=0.4,
                 max_antecedents=250,
                 lexical_dropout= 0.2)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        cuda_device=0 if USE_GPU else -1,
        num_epochs=epochs,
    )    

    metrics = trainer.train()


