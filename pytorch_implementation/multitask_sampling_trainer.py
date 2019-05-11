from typing import List, Dict, Iterable, Any, Set
from collections import defaultdict
import os

import tqdm
import torch
import allennlp
from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.iterators import DataIterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.training import checkpointer
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase

class MultiTaskTrainer(TrainerBase):
    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """
    def __init__(self,
                 serialization_dir: str,
                 task_infos,
                 num_epochs,
                 num_serialized_models_to_keep: int = 10) -> None:
        """
        task1 and task2 should be ditionaries that hold the model, 
        the training, validation, adn test iterator for batches, and 
        the metrics, learning rate, current score, etc for each of the tasks.
        """
        super().__init__(serialization_dir)
        self.task_infos = task_infos
        self.num_epochs = num_epochs
        self.serialization_dir =  serialization_dir
        self.swag_checkpointer= Checkpointer(serialization_dir +"/swag/",
                                         num_serialized_models_to_keep=num_serialized_models_to_keep)
        self.conll_checkpointer = Checkpointer(serialization_dir +"/conll/",
                                             num_serialized_models_to_keep=num_serialized_models_to_keep)

    def save_checkpoint(self, epoch: int) -> None:
        swag_training_state = {"epoch": epoch, "optimizer": self.task_infos["swag"]["optimizer"].state_dict()}
        self.swag_checkpointer.save_checkpoint(epoch, self.task_infos["swag"]["model"].state_dict(), training_state, True)
        swag_training_state = {"epoch": epoch, "optimizer": self.task_infos["lee"]["optimizer"].state_dict()}
        self.conll_checkpointer.save_checkpoint(epoch, self.task_infos["lee"]["model"].state_dict(), training_state, True)

    def restore_model(self, model, current_state, shared_lstm):
        """
        Replace the LSTM parmeters with the one that is being shared
        """
        import pdb; pdb.set_trace()
        if shared_lstm is None:
            # this is the first run of this training session.
            print("shared is None")
            return model
        param_names = list(shared_lstm.state_dict().keys())
        dictionary = shared_lstm.state_dict()
        own_state = model.state_dict()
        for name, param in model.named_parameters():
            before = ''
            if current_state == "lee" and '_context_layer' in name:
                name = name.split('_context_layer.')[1]
                before = '_context_layer.'
            if current_state == 'swag' and 'phrase_encoder' in name:
                name = name.split('phrase_encoder.')[1]
                before = 'phrase_encoder.'
            if name in param_names:
                print("copying parameter with name")
                print(name)
                #TODO: Make sure this is actually copying it to the dictionary 
                own_state[before + name].copy_(dictionary[name])
        model.load_state_dict(own_state)
        return model

    def train(self) -> Dict:
        import numpy as np
        # sample proportionally 
        shared_lstm = None # before trianing, should be rnadomly initialized. 
        current_state = ""
        for epoch in range(0, self.num_epochs):
            swag_train_iter = self.task_infos["swag"]["iterator"](self.task_infos["swag"]["train_data"], num_epochs=1, shuffle=True)
            conll_train_iter = self.task_infos["lee"]["iterator"](self.task_infos["lee"]["train_data"], num_epochs=1, shuffle=True)

            swag_val_iter = self.task_infos["swag"]["iterator"](self.task_infos["swag"]["train_data"], num_epochs=1, shuffle=True)
            conll_val_iter = self.task_infos["lee"]["iterator"](self.task_infos["lee"]["train_data"], num_epochs=1, shuffle=True)


            sampling_ratio = self.task_infos["lee"]["num_train"]/ (self.task_infos["lee"]["num_train"] + self.task_infos["swag"]["num_train"])
            # try smapling_ratio = 0.5 
            sampling_ratio = 0.5
            total_num_batches_train = self.task_infos["lee"]["num_train"] + self.task_infos["swag"]["num_train"]
            #total_num_batches_train = 0
            total_num_batches_lee_val = self.task_infos["lee"]["num_val"]
            total_num_batches_swag_val = self.task_infos["swag"]["num_val"]

            order_list = list(range(100))
            np.random.shuffle(order_list)
 
            total_loss = 0.0
            for i in range(total_num_batches_train):
                index = order_list[i] 
                if index * 0.01 <= sampling_ratio:
                    # thi sis CONLL turn
                    batch_info = self.task_infos["lee"]
                    current_state = "lee"
                else:
                    batch_info = self.task_infos["swag"]
                    current_state = "swag"
                optimizer = batch_info["optimizer"]
                if current_state == "lee":
                    iterator = conll_train_iter
                else:
                    iterator = swag_train_iter
                model = batch_info["model"]
                # TODO: We want to restore the checkpoint here
                # We want to save _just_ the LSTM part from the last round, 
                # and store the rest of the moel. 
                model = self.restore_model(model, current_state, shared_lstm)
                model.train()
                batch = next(iterator, None)
                if batch is None and current_state == "lee":
                    # refill the conll itekrator
                    conll_train_iter = self.task_infos["lee"]["iterator"](self.task_infos["lee"]["train_data"], num_epochs=1, shuffle=True)
                    iterator = conll_train_iter
                optimizer.zero_grad()
                loss = model.forward(**batch)['loss']
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                batch_info["loss"] = loss.item() 
                batch_info["optimizer"] = optimizer
                shared_lstm = model.return_context_layer()
                torch.save(shared_lstm, "shared_lstm")
                torch.save(model, self.serialization_dir +current_state+"current_model")
                batch_info["model"] = model
                print("For this specific task, it has loss"+ str(loss.item()) +"for task "+current_state)
                print("epoch:"+ str(epoch)+ "loss:" +str(total_loss)+"/ ("+str(i+1)+")")

            print("Now validating...")
            batch_info = self.task_infos["lee"]
            iterator = conll_val_iter
            model = batch_info["model"]
            model = self.restore_model(model, "lee", shared_lstm)
            total_lee_val_loss = 0
            model.eval()

            for i in range(total_num_batches_lee_val):
                batch = next(iterator)
                loss = model.forward(**batch)['loss']
                total_lee_val_loss += loss.item()
                batch_info["loss"] = loss.item()
            print("For task:" + current_state + ", epoch val loss"+ str(total_lee_val_loss))


            batch_info = self.task_infos["swag"]
            current_state = "swag"
            iterator = swag_val_iter
            model = batch_info["model"]
            model = self.restore_model(model, "swag", shared_lstm)
            total_swag_val_loss = 0
            model.eval()

            for i in range(500):
                batch = next(iterator)
                loss = model.forward(**batch)['loss']
                total_swag_val_loss += loss.item()
                batch_info["loss"] = loss.item()
            print("For task:" + current_state + ", epoch val loss"+ str(total_swag_val_loss))

            self.save_checkpoint(epoch)

        return {}
