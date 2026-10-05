import sys
import traceback
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

try:
    from src.graph.runner import run_rag
    from tests.evaluation.questions import SAMPLE_QUESTIONS
    from tests.evaluation.metrics import run_eval_suite

    TRACE_DIR = ROOT / "data" / "traces"
    print("Testing question 9 resumption...")
    # Run only 1 question (question 9, index 8) to see what happens
    res = run_eval_suite(
        SAMPLE_QUESTIONS[8:9],
        lambda q: run_rag(q),
        "langgraph",
        pause_seconds=1.0,
        checkpoint_path=TRACE_DIR / "eval_langgraph_checkpoint.csv",
    )
    print("Success! Row count:", len(res))
except Exception as e:
    print("ERROR CAUGHT:")
    traceback.print_exc()
