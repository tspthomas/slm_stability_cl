
# GENERAL
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. "
    "Answer the task exactly as requested. "
    "Return only the final answer."
)

# QWEN
# default prompts as per QWEN documentation - https://huggingface.co/Qwen/Qwen3.5-0.8B
QWEN_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
QWEN_MULTIPLE_CHOICE_PROMPT = "Please show your choice in the answer field with only the choice letter, e.g., \"C\""