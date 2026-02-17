import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantization.qat import QATTrainer, QATConfig
from src.models import ModelManager
from src.data import load_smsa_dataset, load_tweets_dataset


def run_qat_experiment(version_key):
    pass


if __name__ == "__main__":
    pass
