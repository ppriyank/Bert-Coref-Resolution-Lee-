import allennlp

class MultiTaskTrainer(TrainerBase):
    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """
    def __init__(self,
                 serialization_dir: str,
                 task_infos,
                 num_serialized_models_to_keep: int = 10) -> None:
        """
        task1 and task2 should be ditionaries that hold the model, 
        the training, validation, adn test iterator for batches, and 
        the metrics, learning rate, current score, etc for each of the tasks.
        """
        super().__init__(serialization_dir)
        self.task_infos = task_infos
        self.num_epochs = num_epochs
        self.checkpointerSWAG = Checkpointer(serialization_dir +"/swag/",
                                         num_serialized_models_to_keep=num_serialized_models_to_keep)
        self.checkpointerConll = Checkpointer(serialization_dir +"/conll/",
                                             num_serialized_models_to_keep=num_serialized_models_to_keep)

    def save_checkpoint(self, epoch: int) -> None:
        swag_training_state = {"epoch": epoch, "optimizer": self.task_infos["swag"]["optimizer"].state_dict()}
        self.swag_checkpointer.save_checkpoint(epoch, self.task_infos["conll"]["model"].state_dict(), training_state, True)
        swag_training_state = {"epoch": epoch, "optimizer": self.task_infos["conll"]["optimizer"].state_dict()}
        self.swag_checkpointer.save_checkpoint(epoch, self.task_infos["conll"]["model"].state_dict(), training_state, True)

    def restore_model(self, model, epcoh:int):
        """
        Replace the LSTM parmeters with the one that is being shared
        """
        context_layer = model.get_context_layer()
        shared_lstm = torch.load(self.serialization_lstm)
        param_names = list(shared_lstm.named_parameters())
        param_names = [p[0] for p in param_names]
        for name, param in model.named_parameters():
            if name in param_names:
                model.parameters[name] = shared_lstm.model_state[name]

        return model

    def train(self) -> Dict:
        import numpy as np
        swag_train_iter = self.task_infos["swag"]["iterator"](self.task_infos["swag"]["train_data"], \
                                                    num_epochs=1, shuffle=True)
        conll_train_iter = self.task_infos["conll"]["iterator"](self.task_infos["conll"]["train_data"], \
                                                    num_epochs=1, shuffle=True)
        sampling_ratio = conll_train_iter.get_num_batches()/ (conll_train_iter.get_num_batches() + \
                                                                swag_train_iter.get_num_batches())
        total_num_batches = conll_train_iter.get_num_batches() + swag_train_iter.get_num_batches()
        order_list = list(range(100))
        np.random.shuffle(order_list)
        # sample proportionally 
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            for i in range(total_num_batches):
                index = order_list[i] 
                if index * 0.01 <= sampling_ratio:
                    # thi sis CONLL turn
                    batch_info = self.task_infos["conll"]
                else:
                    batch_info = self.task_infos["swag"]
                optimizer = batch_info["optimizer"]
                iterator = batch["iterator"]
                model = batch_info["model"]
                # TODO: We want to restore the checkpoint here
                # We want to save _just_ the LSTM part, and then 
                model = self.restore_model(model)
                model.train()
                batch = next(iterator)
                optimizer.zero_grad()
                loss = model.forward(**batch)['loss']
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                batch["loss"] = loss.item() 
                shared_lstm = model.get_text_embedder()
                torch.save(shared_lstm, "shared_lstm")
                batch["model"] = model
                batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)} for taskname ")er. 

            # Save checkpoint
            self.save_checkpoint(epoch)

        return {}