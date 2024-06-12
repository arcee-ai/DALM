import os


def rank0_print(*args, **kwargs):
    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(*args, **kwargs)
