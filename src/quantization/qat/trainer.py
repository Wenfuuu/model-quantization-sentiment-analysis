class QATTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def prepare_model(self):
        pass

    def train(self, train_loader, val_loader):
        pass

    def evaluate(self, test_loader):
        pass

    def convert_to_quantized(self):
        pass
