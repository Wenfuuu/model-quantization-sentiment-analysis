class QATConfig:
    def __init__(self):
        self.learning_rate = 1e-5
        self.num_epochs = 3
        self.batch_size = 16
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.quantization_backend = "fbgemm"
