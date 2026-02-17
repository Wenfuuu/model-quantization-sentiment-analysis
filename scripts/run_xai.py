import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai import IntegratedGradientsExplainer, LIMEExplainer, SHAPExplainer, OcclusionExplainer
from src.models import ModelManager


def run_xai_analysis(model_path, text):
    pass


if __name__ == "__main__":
    pass
