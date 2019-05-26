// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
{
  "dataset_reader": {
    "type": "coref-bert",
    "token_indexers": {
     "bert": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-base-cased",
            },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    },
    "max_span_width": 10
  },
  "train_data_path": "/beegfs/yp913/dataset/train.english.v4_gold_conll",
  "validation_data_path":"/beegfs/yp913/dataset/dev.english.v4_gold_conll",
  "test_data_path": "/beegfs/yp913/dataset/test.english.v4_gold_conll",
  "model": {
    "type": "coref-bert",
            "text_field_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-base-cased",
              "requires_grad": true
            }
          },
          "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets", "tokens-type-ids"]
          },
          "allow_unmatched_keys": true
        },
    "context_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 768,
        "hidden_size": 200,
        "num_layers": 1
    },
    "mention_feedforward": {
        "input_dim": 1588,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": 4784,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ],
    "lexical_dropout": 0.5,
    "feature_size": 20,
    "max_span_width": 10,
    "spans_per_word": 0.4,
    "max_antecedents": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 150,
    "grad_norm": 5.0,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam"
    }
  }
}
