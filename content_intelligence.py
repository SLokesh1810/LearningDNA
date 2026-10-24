"""LearningDNA Content Intelligence: source-grounded extraction from RAG chunks.

Install: pip install requests python-dotenv
Default provider: existing LearningDNA Colab API. See .env settings below.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
from typing import Any, Protocol
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
load_dotenv()
DEFAULT_CHUNKS = ROOT / "processed" / "chunks.json"
DEFAULT_OUTPUT = ROOT / "content_intelligence"

class LLM(Protocol):
    model: str
    def generate(self, prompt: str, context: str, max_tokens: int) -> str: ...
    def check(self) -> str: ...

class ColabClient:
    def __init__(self):
        base = os.getenv("LEARNINGDNA_LLM_URL", "").strip().rstrip("/")
        self.key = os.getenv("LEARNINGDNA_API_KEY", "").strip()
        if not base or not self.key:
            raise RuntimeError("Set LEARNINGDNA_LLM_URL and LEARNINGDNA_API_KEY in .env")
        self.url = base if base.endswith("/generate") else base + "/generate"
        self.health_url = self.url[:-9] + "/health"
        self.model = "unknown"
    def check(self):
        r = requests.get(self.health_url, timeout=15); r.raise_for_status()
        data = r.json(); self.model = str(data.get("model", "unknown"))
        if data.get("model_loaded") is False: raise RuntimeError("LLM model is not loaded")
        return self.model
    def generate(self, prompt, context, max_tokens):
        r = requests.post(self.url, headers={"X-API-Key": self.key}, json={
            "prompt": prompt, "context": context, "max_new_tokens": min(max_tokens, 600), "temperature": 0.0
        }, timeout=(10, 180))
        if not r.ok: raise RuntimeError(f"LLM API HTTP {r.status_code}: {r.text[:800]}")
        data = r.json(); answer = data.get("response", data.get("answer", ""))
        if not isinstance(answer, str) or not answer.strip(): raise RuntimeError("LLM API returned no text")
        return answer.strip()

class OpenAICompatibleClient:
    def __init__(self):
        base = os.getenv("LEARNINGDNA_LLM_BASE_URL", "").strip().rstrip("/")
        self.key = os.getenv("LEARNINGDNA_LLM_API_KEY", "").strip()
        self.model = os.getenv("LEARNINGDNA_LLM_MODEL", "").strip()
        if not base or not self.model:
            raise RuntimeError("Set LEARNINGDNA_LLM_BASE_URL and LEARNINGDNA_LLM_MODEL in .env")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    def check(self): return self.model
    def generate(self, prompt, context, max_tokens):
        headers = {"Content-Type": "application/json"}
        if self.key: headers["Authorization"] = "Bearer " + self.key
        body = {"model": self.model, "messages": [
            {"role": "system", "content": "Extract source-grounded learning information. Return valid JSON only."},
            {"role": "user", "content": prompt + "\n\nSOURCE CHUNKS:\n" + context}],
            "max_tokens": max_tokens, "temperature": 0}
        r = requests.post(self.url, headers=headers, json=body, timeout=(10, 180))
        if not r.ok: raise RuntimeError(f"OpenAI-compatible API HTTP {r.status_code}: {r.text[:800]}")
        answer = r.json()["choices"][0]["message"]["content"]
        if isinstance(answer, list): answer = "".join(x.get("text", "") for x in answer if isinstance(x, dict))
        if not isinstance(answer, str) or not answer.strip(): raise RuntimeError("LLM API returned no text")
        return answer.strip()

def make_client() -> LLM:
    provider = os.getenv("LEARNINGDNA_LLM_PROVIDER", "custom_api").lower().strip()
    if provider == "custom_api": return ColabClient()
    if provider in {"openai_compatible", "openai-compatible", "local_openai"}: return OpenAICompatibleClient()
    raise RuntimeError(f"Unsupported provider {provider!r}; use custom_api or openai_compatible")

def load_chunks(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        for key in ("chunks", "documents", "data", "items", "records"):
            if isinstance(raw.get(key), list): raw = raw[key]; break
        else: raise ValueError("Expected a list or an object containing a chunks/documents/data list")
    if not isinstance(raw, list): raise ValueError("Chunks file must contain a JSON list")
    chunks, seen = [], set()
    for i, item in enumerate(raw, 1):
        if isinstance(item, str): text, cid, page, meta = item.strip(), str(i), None, {}
        elif isinstance(item, dict):
            text = next((item[k] for k in ("text", "chunk_text", "content", "page_content", "chunk", "body") if item.get(k) is not None), "")
            meta = item.get("metadata", {}); meta = meta if isinstance(meta, dict) else {}
            cid = str(next((item[k] for k in ("chunk_id", "id", "uid") if item.get(k) is not None), i))
            page = next((item[k] for k in ("page", "page_number", "page_num", "source_page", "page_index") if item.get(k) is not None), None)
            if page is None: page = next((meta[k] for k in ("page", "page_number", "page_num", "source_page") if meta.get(k) is not None), None)
            if not text and isinstance(item.get("document"), dict): text = item["document"].get("text", item["document"].get("content", ""))
            text = str(text).strip()
        else: continue
        if not text: continue
        if cid in seen: cid = f"{cid}_dup{i}"
        seen.add(cid); chunks.append({"chunk_id": cid, "text": text, "page": page, "metadata": meta})
    if not chunks: raise ValueError("No chunks with usable text found")
    return chunks

def _escape_controls(text: str) -> str:
    out, inside, escaped = [], False, False
    for ch in text:
        if inside:
            if escaped: out.append(ch); escaped = False
            elif ch == "\\": out.append(ch); escaped = True
            elif ch == '"': out.append(ch); inside = False
            elif ord(ch) < 32: out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, f"\\u{ord(ch):04x}"))
            else: out.append(ch)
        else:
            out.append(ch)
            if ch == '"': inside = True
    return "".join(out)

def parse_json(text: str) -> dict[str, Any]:
    """Parse JSON and repair only common, low-risk small-model formatting mistakes."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    start = text.find("{")
    if start < 0: raise ValueError("No JSON object found in model response")
    # Find the first balanced object without mistaking braces inside strings.
    depth = 0; inside = escaped = False; end = None
    for i in range(start, len(text)):
        ch = text[i]
        if inside:
            if escaped: escaped = False
            elif ch == "\\": escaped = True
            elif ch == '"': inside = False
        else:
            if ch == '"': inside = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i + 1; break
    candidate = text[start:end] if end else text[start:]
    variants = [candidate, _escape_controls(candidate)]
    # Phi-3 sometimes emits a bare numeric confidence (e.g. `,\n0.0\n}`)
    # instead of a key/value. Drop that orphan only when it cannot be valid JSON.
    variants.append(re.sub(r"(?:,\s*|(?<=\")\s*\n\s*)-?\d+(?:\.\d+)?\s*(?=[}\]])", "", _escape_controls(candidate)))
    variants.append(re.sub(r",\s*([}\]])", r"\1", _escape_controls(candidate)))
    errors = []
    for variant in variants:
        variant = re.sub(r",\s*([}\]])", r"\1", variant)
        try:
            obj = json.loads(variant)
            if isinstance(obj, dict): return obj
        except json.JSONDecodeError as e: errors.append(f"{e.msg} at line {e.lineno}, column {e.colno}")
    raise ValueError("Model response is not valid JSON (" + (errors[-1] if errors else "invalid object") + ")")

PROMPT = '''Extract information from the supplied chunks only. Return ONE valid compact JSON object and nothing else, using this exact structure:
{"concepts":[{"name":"...","definition":"...","aliases":[],"evidence":[{"chunk_id":"...","quote":"..."}]}],"relationships":[{"source_concept":"...","target_concept":"...","relationship_type":"prerequisite|is_a_type_of|part_of|contrasts_with|used_for|related_to","evidence_type":"explicit|inferred","evidence":"...","chunk_ids":[]}],"learning_objectives":[{"objective":"...","related_concepts":[],"bloom_level":"remember|understand|apply|analyze|evaluate|create","evidence_chunk_ids":[]}]}
Rules: at most 3 concepts, 1 relationship, and 2 objectives. Keep definitions and quotes short. Quotes must be exact substrings from the given chunk. Use only supplied chunk IDs. Do not output page numbers or confidence scores. Do not invent unsupported details. Use [] when no supported items exist. JSON strings must not contain literal line breaks. No markdown.'''

def extract(client: LLM, batch: list[dict[str, Any]], token_cap: int, diagnostics: Path) -> dict[str, Any]:
    # Source text appears exactly once in the request to avoid consuming context twice.
    context = "\n\n--- CHUNK ---\n\n".join(
        f"[chunk_id={c['chunk_id']} | page={c['page'] if c['page'] is not None else 'null'}]\n{c['text']}" for c in batch)
    raw = client.generate(PROMPT, context, token_cap)
    try: result = parse_json(raw)
    except ValueError as first:
        # Keep the exact response; this is useful if a model/provider behaves differently.
        diagnostics.mkdir(parents=True, exist_ok=True)
        raw_path = diagnostics / f"batch_response_{hashlib.sha1(context.encode()).hexdigest()[:10]}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        # A single shorter retry only for invalid output; no unbounded retries.
        retry_prompt = PROMPT + "\nYour previous output was invalid. Return a SHORTER, complete JSON object. No extra text."
        raw2 = client.generate(retry_prompt, context, max(180, min(token_cap, 400)))
        try: result = parse_json(raw2)
        except ValueError as second:
            retry_path = diagnostics / f"batch_retry_response_{hashlib.sha1(context.encode()).hexdigest()[:10]}.txt"
            retry_path.write_text(raw2, encoding="utf-8")
            raise ValueError(f"Invalid JSON after one retry. First: {first}; second: {second}. Raw responses saved in {diagnostics}.") from second
    for key in ("concepts", "relationships", "learning_objectives"):
        if not isinstance(result.get(key), list): result[key] = []
    return result

def norm(s: Any) -> str: return re.sub(r"\s+", " ", str(s).casefold()).strip()
def strings(value: Any) -> list[str]:
    if isinstance(value, str): value = [value]
    if not isinstance(value, list): return []
    return list(dict.fromkeys(str(x).strip() for x in value if x is not None and str(x).strip()))
def save(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)

def merge(results: list[dict[str, Any]], chunks: list[dict[str, Any]]):
    source = {c["chunk_id"]: c for c in chunks}; concepts, rels, objs = {}, {}, {}
    for result in results:
        for x in result.get("concepts", []):
            if not isinstance(x, dict) or not str(x.get("name", "")).strip(): continue
            name = str(x["name"]).strip(); key = norm(name); evidence = []
            for ev in x.get("evidence", []) if isinstance(x.get("evidence"), list) else []:
                if not isinstance(ev, dict): continue
                cid, quote = str(ev.get("chunk_id", "")), str(ev.get("quote", "")).strip()
                if cid in source and quote and quote.casefold() in source[cid]["text"].casefold():
                    evidence.append({"chunk_id": cid, "page": source[cid]["page"], "quote": quote})
            rec = {"name": name, "definition": str(x.get("definition", "")).strip(), "aliases": strings(x.get("aliases")), "evidence": evidence}
            if key not in concepts: concepts[key] = rec
            else:
                old = concepts[key]
                if len(rec["definition"]) > len(old["definition"]): old["definition"] = rec["definition"]
                old["aliases"] = list(dict.fromkeys(old["aliases"] + rec["aliases"]))
                known = {(e["chunk_id"], e["quote"]) for e in old["evidence"]}
                old["evidence"].extend(e for e in evidence if (e["chunk_id"], e["quote"]) not in known)
        for x in result.get("relationships", []):
            if not isinstance(x, dict): continue
            a, b = str(x.get("source_concept", "")).strip(), str(x.get("target_concept", "")).strip()
            typ = str(x.get("relationship_type", "related_to")).strip().lower()
            if not a or not b or norm(a) == norm(b) or typ not in {"prerequisite", "is_a_type_of", "part_of", "contrasts_with", "used_for", "related_to"}: continue
            rec = {"source_concept": a, "target_concept": b, "relationship_type": typ,
                   "evidence_type": "explicit" if x.get("evidence_type") == "explicit" else "inferred",
                   "evidence": str(x.get("evidence", "")).strip(), "chunk_ids": [i for i in strings(x.get("chunk_ids")) if i in source]}
            rels[(norm(a), norm(b), typ)] = rec
        for x in result.get("learning_objectives", []):
            if not isinstance(x, dict) or not str(x.get("objective", "")).strip(): continue
            objective = str(x["objective"]).strip(); key = norm(objective)
            bloom = str(x.get("bloom_level", "understand")).lower()
            if bloom not in {"remember", "understand", "apply", "analyze", "evaluate", "create"}: bloom = "understand"
            rec = {"objective": objective, "related_concepts": strings(x.get("related_concepts")), "bloom_level": bloom,
                   "evidence_chunk_ids": [i for i in strings(x.get("evidence_chunk_ids")) if i in source]}
            if key not in objs: objs[key] = rec
    concept_list = sorted(concepts.values(), key=lambda x: x["name"].casefold()); idmap = {}
    for i, c in enumerate(concept_list, 1):
        c["concept_id"] = f"CON_{i:04d}"; idmap[norm(c["name"])] = c["concept_id"]
        c["source_pages"] = list(dict.fromkeys(e["page"] for e in c["evidence"] if e["page"] is not None))
    rel_list = []
    for r in rels.values():
        if norm(r["source_concept"]) in idmap and norm(r["target_concept"]) in idmap:
            r["source_concept_id"] = idmap[norm(r["source_concept"])]
            r["target_concept_id"] = idmap[norm(r["target_concept"])]
            rel_list.append(r)
    return concept_list, rel_list, sorted(objs.values(), key=lambda x: x["objective"].casefold())

def main() -> int:
    ap = argparse.ArgumentParser(description="Extract source-grounded concepts, relationships and objectives.")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    ap.add_argument("--max-new-tokens", type=int, default=560)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.batch_size < 1 or args.max_batches < 0 or not 1 <= args.max_new_tokens <= 600: ap.error("Use batch-size >= 1, max-batches >= 0 and max-new-tokens between 1 and 600")
    if not args.chunks.exists(): print(f"[ERROR] Chunks file not found: {args.chunks}", file=sys.stderr); return 2
    try:
        client = make_client(); model = client.check()
        provider = os.getenv("LEARNINGDNA_LLM_PROVIDER", "custom_api")
        print(f"[OK] LLM provider ready | Provider: {provider} | Model: {model}")
        chunks = load_chunks(args.chunks); batches = [chunks[i:i+args.batch_size] for i in range(0, len(chunks), args.batch_size)]
        if args.max_batches: batches = batches[:args.max_batches]
        print(f"[OK] Loaded {len(chunks)} chunks | Processing {sum(map(len,batches))} chunks in {len(batches)} batches")
        sig = hashlib.sha256(args.chunks.read_bytes()).hexdigest()
        checkpoint_path = args.output_dir / "content_intelligence_checkpoint.json"
        results = [None] * len(batches)
        if checkpoint_path.exists() and not args.no_resume:
            try:
                old = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if old.get("source_sha256") == sig and old.get("batch_size") == args.batch_size and old.get("model") == model:
                    prior = old.get("results", [])
                    for i, r in enumerate(prior[:len(results)]):
                        if isinstance(r, dict) and not r.get("_error"): results[i] = r
                    print(f"[INFO] Reusing {sum(r is not None for r in results)} completed batch(es)")
            except (OSError, json.JSONDecodeError): pass
        failures = []
        for i, batch in enumerate(batches):
            if results[i] is not None: print(f"[SKIP] Batch {i+1}/{len(batches)} already complete"); continue
            print(f"\n[INFO] Batch {i+1}/{len(batches)} ({len(batch)} chunks)")
            try:
                result = extract(client, batch, args.max_new_tokens, args.output_dir / "diagnostics"); results[i] = result
                print(f"[OK] {len(result['concepts'])} concepts, {len(result['relationships'])} relationships, {len(result['learning_objectives'])} objectives")
            except Exception as e:
                results[i] = {"concepts": [], "relationships": [], "learning_objectives": [], "_error": str(e)}
                failures.append({"batch": i+1, "error": str(e)})
                diag = args.output_dir / "diagnostics" / f"batch_{i+1:04d}_error.txt"
                diag.parent.mkdir(parents=True, exist_ok=True); diag.write_text(str(e), encoding="utf-8")
                print(f"[ERROR] Batch {i+1}: {e}")
            save(checkpoint_path, {"source_sha256": sig, "batch_size": args.batch_size, "model": model, "results": results})
        completed_chunks = [c for b in batches for c in b]
        concepts, relationships, objectives = merge([r or {} for r in results], completed_chunks)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save(args.output_dir / "concept_catalog.json", {"concepts": concepts})
        save(args.output_dir / "concept_relationships.json", {"relationships": relationships})
        save(args.output_dir / "learning_objectives.json", {"learning_objectives": objectives})
        report = {"source_file": str(args.chunks.resolve()), "provider": provider, "model": model,
                  "statistics": {"chunks_processed": len(completed_chunks), "concepts": len(concepts), "relationships": len(relationships), "learning_objectives": len(objectives), "failed_batches": len(failures)},
                  "failed_batches": failures,
                  "quality_notes": ["Evidence quotes are retained only when they exactly match source chunk text (case-insensitive).", "Review inferred prerequisites before using them to control learner progression.", "LLM-generated output requires human quality review."]}
        save(args.output_dir / "content_intelligence_report.json", report)
        print("\nLEARNINGDNA CONTENT INTELLIGENCE RESULTS")
        print(f"Chunks processed        : {len(completed_chunks)}\nConcepts extracted       : {len(concepts)}\nRelationships extracted  : {len(relationships)}\nLearning objectives      : {len(objectives)}\nFailed batches           : {len(failures)}\nOutput directory         : {args.output_dir.resolve()}")
        return 1 if failures else 0
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())