"""An implementation of a model for the SWAG dataset using AllenNLP."""

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional)

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2VecEncoder,
    SimilarityFunction,
    TextFieldEmbedder)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SWAGExampleModel(Model):
    """An example model for the SWAG dataset.
    This model predicts on the SWAG task by encoding the startphrase and
    the four endings, taking their dot products and then predicting the
    most similar one.
    Parameters
    ----------
    vocab : Vocabulary
        The vocabulary for the data.
    text_field_embedder : TextFieldEmbedder
        A module to embed the text for both the startphrase and the
        endings.
    startphrase_encoder : Seq2VecEncoder
        The encoder for the startphrase.
    ending_encoder : Seq2VecEncoder
        The encoder for the endings. It will be applied to each ending
        separately.
    similarity : SimilarityFunction
        The notion of similarity to use between the startphrase and the
        ending embeddings.
    initializer : InitializerApplicator
        An initializer defining how to initialize all variables.
    regularizer : RegularizerApplicator, optional (default=None)
        Regularization to apply for training.
    """
    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            phrase_encoder: Seq2SeqEncoder, # this is the BiLSTM
    ) -> None:
        super().__init__(vocab)

        # validate the configuration
        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            phrase_encoder.get_input_dim(),
            "text field embedding dim",
            "startphrase encoder input dim")
        # bind all attributes to the instance
        self.text_field_embedder = text_field_embedder.cuda()
        self.phrase_encoder = phrase_encoder.cuda()
        # set the training and validation losses
        self.xentropy = torch.nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    def return_context_layer(self):
        return self.phrase_encoder
    def forward(
            self,
            startphrase: Dict[str, torch.LongTensor],
            ending0: Dict[str, torch.LongTensor],
            ending1: Dict[str, torch.LongTensor],
            ending2: Dict[str, torch.LongTensor],
            ending3: Dict[str, torch.LongTensor],
            label: Optional[torch.IntTensor] = None,
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for predicting the best ending.
        Parameters
        ----------
        startphrase : Dict[str, torch.LongTensor]
            The startphrase field.
        ending0 : Dict[str, torch.LongTensor]
            The ending0 field.
        ending1 : Dict[str, torch.LongTensor]
            The ending1 field.
        ending2 : Dict[str, torch.LongTensor]
            The ending2 field.
        ending3 : Dict[str, torch.LongTensor]
            The ending3 field.
        label : Optional[torch.IntTensor]
            The index of the correct ending.
        metadata : Optional[List[Dict[str, Any]]]
            Optional additional metadata.
        Returns
        -------
        A dictionary containing:
        logits : torch.FloatTensor
            A batch_size x num_endings tensor giving the logit for each
            of the endings.
        probabilities : torch.FloatTensor
            A batch_size x num_endings tensor giving the probabilities
            for each ending.
        loss : Optional[torch.FloatTensor]
            The training loss.
        """
        # pass the startphrase and endings through the initial text
        # embedding
        startphrase_initial = self.text_field_embedder(startphrase)
        ending0_initial = self.text_field_embedder(ending0)
        ending1_initial = self.text_field_embedder(ending1)
        ending2_initial = self.text_field_embedder(ending2)
        ending3_initial = self.text_field_embedder(ending3)
        # embed the startphrase and endings
        startphrase_embedding = self.phrase_encoder(
            startphrase_initial,
            get_text_field_mask(startphrase))
        ending0_embedding = self.phrase_encoder(ending0_initial,get_text_field_mask(ending0))
        ending1_embedding = self.phrase_encoder(ending1_initial,get_text_field_mask(ending1))
        ending2_embedding = self.phrase_encoder(
            ending2_initial,
            get_text_field_mask(ending2))
        ending3_embedding = self.phrase_encoder(
            ending3_initial,
            get_text_field_mask(ending3))
        # get a linear layer that projects it down to
        # take the dot product of the embeddings

        # first, stack the endings so that we get a batch x num_endings
        # x embedding_dim tensor, then add an extra dimension to the
        # startphrase batch so it's a batch x embedding_dim x 1 tensor,
        # and broadcast matrix multiplication across the last two
        # dimensions to get the dot products
        logits = torch.stack(
            [
                ending0_embedding,
                ending1_embedding,
                ending2_embedding,
                ending3_embedding
            ],
            dim=-2
        ).bmm(
            startphrase_embedding.unsqueeze(-1)
        ).squeeze()

        # compute the probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # compute the loss
        if label is not None:
            loss = self.xentropy(logits, label.long().view(-1))
            self.accuracy(logits, label)
        else:
            loss = None

        # return the output
        return {
            'logits': logits,
            'probabilities': probabilities,
            'loss': loss
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}
