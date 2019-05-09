


class MultiTaskTrainer(TrainerBase):
    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """
    def __init__(self,
                 model: Model,
                 serialization_dir: str,
                 task1, 
                 task2,
                 num_serialized_models_to_keep: int = 10) -> None:
        """
        task1 and task2 should be ditionaries that hold the model, 
        the training, validation, adn test iterator for batches, and 
        the metrics, learning rate, current score, etc for each of the tasks.
        """
        super().__init__(serialization_dir)
        self.task1 = {}
        self.taask2 = {}
        self.iterator = iterator
        self.mingler = mingler
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.checkpointer = Checkpointer(serialization_dir,
                                         num_serialized_models_to_keep=num_serialized_models_to_keep)

    def save_checkpoint(self, epoch: int) -> None:
        training_state = {"epoch": epoch, "optimizer": self.optimizer.state_dict()}
        self.checkpointer.save_checkpoint(epoch, self.model.state_dict(), training_state, True)

    def restore_checkpoint(self) -> int:
        model_state, trainer_state = self.checkpointer.restore_checkpoint()
        if not model_state and not trainer_state:
            return 0
        else:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(trainer_state["optimizer"])
            return trainer_state["epoch"] + 1


    def train(self) -> Dict:
        start_epoch = self.restore_checkpoint()

        self.model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            batches = self.iterator()
            tqdm.tqdm(self.iterator(self.mingler.mingle(self.datasets), num_epochs=1))
            for i, batch in enumerate(batches):
                self.optimizer.zero_grad()
                loss = self.model.forward(**batch)['loss']
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
                batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)}")

            # Save checkpoint
            self.save_checkpoint(epoch)

        return {}