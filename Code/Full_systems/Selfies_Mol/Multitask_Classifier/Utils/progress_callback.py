from tqdm.notebook import tqdm

class ProgressBarCallback:
    def __init__(self, steps_per_epoch, total_epochs, return_epochs=True, model_name=None):
        """
        Initialize the progress bar callback.
        """
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.current_step = 0
        self.current_epoch = 0
        self.pbar = None  # Placeholder for the progress bar
        self.return_epochs = return_epochs # if the completed epoch number needs to be printed
        self.model_name = model_name

    def __call__(self, model, step):
        """
        Update the progress bar at each step.
        """
        if self.pbar is None:  # Initialize the progress bar at the start
            if self.model_name is None:
                self.pbar = tqdm(total=self.steps_per_epoch * self.total_epochs, desc="Training Progress", unit="step")
            else:
                self.pbar = tqdm(total=self.steps_per_epoch * self.total_epochs, desc=f"Training Model {self.model_name}", unit="step")

        # Update the progress bar
        self.pbar.update(1)
        self.current_step += 1

        # Check if an epoch is completed
        if self.return_epochs and  self.current_step % self.steps_per_epoch == 0:
            self.current_epoch += 1
            print(f"Epoch {self.current_epoch}/{self.total_epochs} completed.")

    def close(self):
        """
        Close the progress bar when training is done.
        """
        if self.pbar is not None:
            self.pbar.close()