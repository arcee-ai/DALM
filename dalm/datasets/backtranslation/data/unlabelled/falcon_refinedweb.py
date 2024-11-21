import json

from datasets import load_dataset
from tqdm import tqdm


def is_clean_text(text: str) -> bool:
    # according to Table 2, the response lengths of A_{5}^{(2)} range from 1047 to 2279
    if len(text) < 1000 or len(text) > 2200:
        return False

    segs = text.split("\n")
    if any(len(seg) < 5 for seg in segs):
        return False

    return True


def main():
    num_samples = 502000
    dump_filepath = "data/unlabelled/falcon-refinedweb-sampled.jsonl"

    ds = load_dataset("tiiuae/falcon-refinedweb", streaming=True, split="train")
    fout = open(dump_filepath, "a")
    cnt = 0
    tot = 0
    pbar = tqdm(total=num_samples)
    for ins in ds:
        if cnt >= num_samples:
            break
        if is_clean_text(ins["content"]):
            ins["timestamp"] = ins["timestamp"].strftime("%Y%m%d%H%M%S")
            ins_str = json.dumps(ins, ensure_ascii=False)
            fout.write(f"{ins_str}\n")
            fout.flush()
            cnt += 1
            pbar.update(1)
        tot += 1
        pbar.set_postfix({"tot": tot, "valid": cnt})
    fout.close()


if __name__ == "__main__":
    main()
