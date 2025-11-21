import pandas as pd
import json
import hashlib
from tqdm import tqdm
import os, json, hashlib, time
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
from dataclass import EHRRecord, SOAP

# params
INPUT_CSV = "emergency_nursing_with_maincol_nocovid.csv"
OUT_JSONL = "ehr_structured.jsonl"
ID_COL = "病歷編號"
TEXT_PARTS = ["主訴(S)", "客觀(O)", "診斷(A)"]
ICD_DESC_COLS = ["ICD101", "ICD102", "ICD103", "ICD104", "ICD105", "ICD106"]
INPUT_JSONL = "ehr_structured.jsonl"
OUTPUT_DIR = "faiss_index_flat_gpu"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512
BATCH_SIZE = 16384
TOKEN_BATCH = 2048

usecols = [ID_COL] + TEXT_PARTS + ICD_DESC_COLS
df = pd.read_csv(INPUT_CSV, dtype=str, usecols=usecols, encoding="utf-8-sig", low_memory=False)
df = df.fillna("").astype(str)


def anon(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]


def record_builder(row):
    icd_codes = [row[c] for c in ICD_DESC_COLS if row[c]]
    soap = SOAP(
        S=row["主訴(S)"] or None, O=row["客觀(O)"] or None, A=row["診斷(A)"] or None, P=None
    )

    record = EHRRecord(
        id=anon(row[ID_COL]),
        visit_date=None,
        department=None,
        sex=None,
        visit_type=None,
        assessment=None,
        history=None,
        icd_codes=icd_codes,
        soap=soap,
    )
    return record


tqdm.pandas(desc="fill EHR records")
records = df.progress_apply(record_builder, axis=1).tolist()

records = [r for r in records if any([r.soap.S, r.soap.O, r.soap.A])]
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(rec.to_json() + "\n")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSONL
ids, texts = [], []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading lines"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        _id = str(obj.get("id", ""))
        _tx = str(obj.get("text", "")).strip()
        if not _id or not _tx:
            continue
        ids.append(_id)
        texts.append(_tx)
print(f"[LOAD] Records: {len(texts)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(MODEL_NAME, device=device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# chunk
chunked_texts, chunk_parents = [], []

for i in tqdm(range(0, len(texts), TOKEN_BATCH), desc="Batch tokenizing"):
    sub_texts = texts[i : i + TOKEN_BATCH]
    sub_ids = ids[i : i + TOKEN_BATCH]

    encodings = tokenizer.batch_encode_plus(
        sub_texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    for rid, input_ids in enumerate(encodings["input_ids"]):
        if not input_ids:
            continue
        parent_id = sub_ids[rid]
        for j in range(0, len(input_ids), MAX_TOKENS):
            sub_chunk = input_ids[j : j + MAX_TOKENS]
            if not sub_chunk:
                continue
            chunked_texts.append(tokenizer.decode(sub_chunk, skip_special_tokens=True))
            chunk_parents.append(parent_id)

if len(chunked_texts) == 0:
    raise RuntimeError("No chunks generated. Check MAX_TOKENS or input content.")

start_total = time.time()
index = None
dim = None
res = faiss.StandardGpuResources()
pbar = tqdm(range(0, len(chunked_texts), BATCH_SIZE), desc="Embedding batches", unit="batch")

for start in pbar:
    t0 = time.time()
    batch = chunked_texts[start : start + BATCH_SIZE]

    # GPU embedding
    vec = model.encode(
        batch,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    if index is None:
        dim = vec.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # Add vectors directly on GPU
    index.add(vec)

    elapsed = time.time() - t0
    pbar.set_postfix({"batch_time(s)": f"{elapsed:.2f}", "total": index.ntotal})

# Save FAISS index to disk
cpu_index_final = faiss.index_gpu_to_cpu(index)
faiss.write_index(cpu_index_final, os.path.join(OUTPUT_DIR, "index.faiss"))