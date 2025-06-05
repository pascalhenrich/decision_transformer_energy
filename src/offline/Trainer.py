class Trainer():
    def __init__(self, model, dataset):
        self.model = model

    def train_iteration(self, num_steps, curr_iter):
        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()

        self.model.eval()
        # Eval


    def train_step():
