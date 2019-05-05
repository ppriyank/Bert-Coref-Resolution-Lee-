
import torch 
import allennlp
from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader
import allennlp.data.token_indexers.wordpiece_indexer.PretrainedBertIndexer
from allennlp_coref import CoreferenceResolver


def main():
	# load datasetreader 

