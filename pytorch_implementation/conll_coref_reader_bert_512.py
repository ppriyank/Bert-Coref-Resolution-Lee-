import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, MetadataField, SequenceLabelField
from span_field import SpanField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import copy

def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


@DatasetReader.register("coref-bert")
class ConllCorefBertReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.
    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.
    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.token_indexer = token_indexer = PretrainedBertIndexer("bert-base-cased", do_lowercase=False)
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)
            new_sentences = [s.words for s in sentences]
            flattened_sentences = [self._normalize_word(word) for sentence in new_sentences for word in sentence]
            def tokenizer(s: str):
                    return self.token_indexer.wordpiece_tokenizer(s)
            flattened_sentences = tokenizer(" ".join(flattened_sentences))
            yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

    def align_token(self, text, span):
        """
        Retokenize one span for one individual span.
        """
        current = self.token_indexer.wordpiece_tokenizer(" ".join(text[:span[0]]))
        start_span = len(current)
        span_embedding = self.token_indexer.wordpiece_tokenizer(" ".join(text[span[0]:span[1]]))
        end_span = start_span + len(span_embedding)
        return start_span, end_span

    def align_clusters_to_tokens(self, text, clusters):
        new_clusters = []
        for cluster in clusters:
            new_cluster = []
            for span in cluster:
                new_cluster.append(self.align_token(text, span))
            new_clusters.append(new_cluster)
        return new_clusters

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.
        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        flattened_sentences = [self._normalize_word(word) for sentence in sentences for word in sentence]
        # align clusters
        gold_clusters = self.align_clusters_to_tokens(flattened_sentences, gold_clusters)
        def tokenizer(s: str):
            return self.token_indexer.wordpiece_tokenizer(s)

        flattened_sentences = tokenizer(" ".join(flattened_sentences))
        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
        if len(flattened_sentences) > 512:
            #import pdb
            #pdb.set_trace()
            text_field = TextField([Token(["[CLS]"])] + [Token(word) for word in flattened_sentences[:512]]+ [Token(["[SEP]"])] , self._token_indexers)
            total_list = [text_field]
            import math
            for i in range(math.ceil(float(len(flattened_sentences[512:]))/100.0)):
                # slide by 100
                text_field = TextField([Token(["[CLS]"])] + [Token(word) for word in flattened_sentences[512+ (i*100): 512 + ((i+1) * 100 )]]+ [Token(["[SEP]"])] , self._token_indexers)
                total_list.append(text_field)
            text_field = ListField(total_list)
            # doing the Listfield 

        else:
            text_field = TextField([Token(["[CLS]"])] + [Token(word) for word in flattened_sentences]+ [Token(["[SEP]"])] , self._token_indexers) 
        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None
        sentence_offset = 0
        normal = []
        for sentence in sentences:
            # enumerate the spans.
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if span_labels is not None:
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)
                # align the spans to the BERT tokeniation
                normal.append((start, end))
                # span field for Span, which needs to be a flattened esnetnece.
                span_field = None
                """
                if len(flattened_sentences) > 512:
                    span_field = TextField([Token(["[CLS]"])] + [Token(word) for word in flattened_sentences]+ [Token(["[SEP]"])] , self._token_indexers) 
                else:
                    span_field = text_field
                """
                spans.append(SpanField(start, end, span_field))
            sentence_offset += len(sentence)

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)
        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
