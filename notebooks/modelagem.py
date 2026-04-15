# %%
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext
from src.modeling.experiment_runner import run_classification_experiments

context = PipelineContext.from_notebook(__file__)
run_classification_experiments(context.root_dir, logger=context.logger)
