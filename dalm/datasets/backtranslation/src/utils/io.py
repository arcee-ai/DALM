import json
from pathlib import Path

import transformers


def dump_jsonlines(obj, filepath, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = json.dumps(d, ensure_ascii=False, **kwargs)
            fout.write("{}\n".format(line_d))


def load_jsonlines(filepath, **kwargs):
    data = list()
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            line_data = json.loads(line.strip())
            data.append(line_data)
    return data


def load_json(filepath, **kwargs):
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin, **kwargs)
    return data


def dump_json(obj, filepath, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wt", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, **kwargs)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
