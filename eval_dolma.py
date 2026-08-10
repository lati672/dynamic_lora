"""Token-weighted perplexity evaluation on a reproducible Dolma subset."""
import argparse, json, math, os, time
from pathlib import Path
from typing import Any
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset-id", default="allenai/dolma")
    p.add_argument("--dataset-config", default="v1_6-sample")
    p.add_argument("--num-documents", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    p.add_argument("--max-length", type=int, default=2_048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=("auto","float32","float16","bfloat16"), default="auto")
    p.add_argument("--output", type=Path, default=Path("outputs/dolma/base_20k.json"))
    p.add_argument("--token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--log-every", type=int, default=100)
    a = p.parse_args()
    if min(a.num_documents,a.shuffle_buffer_size,a.max_length,a.batch_size) <= 0:
        p.error("document, buffer, length, and batch values must be positive")
    return a

def dtype_for(name):
    if name != "auto": return getattr(torch, name)
    if not torch.cuda.is_available(): return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def main():
    a = arguments(); torch.manual_seed(a.seed); dtype = dtype_for(a.dtype)
    tok = AutoTokenizer.from_pretrained(a.model_id, token=a.token)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    kw: dict[str, Any] = {"token": a.token, "dtype": dtype}
    if torch.cuda.is_available(): kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model_id, **kw).eval()
    device = next(model.parameters()).device
    manifest = hf_hub_download(
        repo_id=a.dataset_id, filename=f"urls/{a.dataset_config}.txt",
        repo_type="dataset", token=a.token,
    )
    shard_urls = [url.strip() for url in Path(manifest).read_text().splitlines() if url.strip()]
    if not shard_urls:
        raise RuntimeError(f"Dolma URL manifest is empty: {manifest}")
    stream = load_dataset(
        "json", data_files={"train": shard_urls}, split="train",
        streaming=True, token=a.token,
    )
    stream = stream.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer_size)
    iterator = (r for r in stream if str(r.get("text","")).strip())
    nll = 0.0; tokens = 0; seen = 0; ids = []; sources = {}; started = time.time()
    with torch.inference_mode():
        while seen < a.num_documents:
            batch = []
            try:
                for _ in range(min(a.batch_size, a.num_documents-seen)):
                    batch.append(next(iterator))
            except StopIteration:
                raise RuntimeError(f"Dolma ended after {seen+len(batch)} documents")
            enc = tok([r["text"] for r in batch], padding=True, truncation=True,
                      max_length=a.max_length, return_tensors="pt")
            x = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
            labels = x.masked_fill(mask == 0, -100)
            count = int((labels[:,1:] != -100).sum())
            if count:
                loss = model(input_ids=x, attention_mask=mask, labels=labels).loss
                nll += float(loss) * count; tokens += count
            seen += len(batch); ids.extend(str(r.get("id","")) for r in batch)
            for r in batch:
                source = str(r.get("source","")) or "unknown"
                sources[source] = sources.get(source,0) + 1
            if a.log_every and math.ceil(seen/a.batch_size) % a.log_every == 0:
                loss = nll/tokens
                print(f"documents={seen:,}/{a.num_documents:,} tokens={tokens:,} "
                      f"loss={loss:.6f} ppl={math.exp(loss):.4f}", flush=True)
    loss = nll/tokens
    result = {"model_id":a.model_id,"dataset_id":a.dataset_id,
      "dataset_config":a.dataset_config,"num_documents":seen,
      "num_predicted_tokens":tokens,"max_length":a.max_length,
      "seed":a.seed,"shuffle_buffer_size":a.shuffle_buffer_size,
      "dtype":str(dtype).removeprefix("torch."),"mean_nll":loss,
      "perplexity":math.exp(loss),"elapsed_seconds":time.time()-started,
      "source_counts":dict(sorted(sources.items())),"sampled_document_ids":ids}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="sampled_document_ids"},indent=2))
    print(f"Wrote {a.output}")

if __name__ == "__main__": main()
