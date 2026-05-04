"""
This module defines constants used throughout the code.
"""

# GENERAL
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. "
    "Answer the task exactly as requested. "
    "Return only the final answer."
)

GENERATION_TEMPERATURE = 0.0
GENERATION_TOP_P = 1.0
GENERATION_DO_SAMPLE = False
GENERATION_USE_CACHE = True

# QWEN
# default prompts as per QWEN documentation - https://huggingface.co/Qwen/Qwen3.5-0.8B
QWEN_MATH_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
QWEN_MULTIPLE_CHOICE_PROMPT = (
    'Please show your choice in the answer field with only the choice letter, e.g., "C"'
)


# TASKS
TASK_TRACE_SCIENCEQA = "scienceqa"
TASK_TRACE_FOMC = "fomc"
TASK_TRACE_NUMGLUECM = "numgluecm"


TASK_PROMPT_MAP = {
    TASK_TRACE_SCIENCEQA: QWEN_MULTIPLE_CHOICE_PROMPT,
    TASK_TRACE_FOMC: QWEN_MULTIPLE_CHOICE_PROMPT,
    TASK_TRACE_NUMGLUECM: QWEN_MATH_PROMPT,
}

# TASKS
MULTIPLE_CHOICE_TASKS = [TASK_TRACE_SCIENCEQA, TASK_TRACE_FOMC]
NUMERIC_TASKS = [TASK_TRACE_NUMGLUECM]


# LORA
LORA_RANK = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = "all-linear"
