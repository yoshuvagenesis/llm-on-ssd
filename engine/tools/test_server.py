"""End-to-end test of the chat server (--serve) on a tiny random model.

  python3 tools/test_server.py [path/to/moe-engine]

Builds a small byte-level tokenizer and a tiny model with the same vocabulary, starts the
server and checks: the page, /v1/models, /api/stats, greedy answers equal the terminal
mode, streaming equals non-streaming, conversation reuse gives the same answer as a cold
start, queueing of parallel requests, stopping mid-answer, dropping old messages when the
context is full, thinking mode, and the password.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "trace"))
import qwen35_ref  # noqa: E402
import selftest  # noqa: E402
import test_tokenizer as tt  # noqa: E402

ENGINE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "target", "release", "moe-engine")
FAILS = []


def check(name, ok, detail=""):
    print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def build(d, arch="qwen3"):
    corpus = open(os.path.join(HERE, "..", "..", "trace", "corpus.txt"), encoding="utf-8").read()
    merges = tt.train(tt.TEXTS + [corpus[:8000]], 600)
    vocab = {tt.B2U[b]: b for b in range(256)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    sp = {s: len(vocab) + i for i, s in enumerate(special)}
    tj = {
        "added_tokens": [{"id": i, "content": s, "special": True} for s, i in sp.items()],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": tt.QWEN35 if arch == "qwen35" else tt.QWEN},
             "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel"}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": [[a, b] for a, b in merges]},
    }
    json.dump(tj, open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    V = len(vocab) + len(special)
    V += (-V) % 32
    if arch == "qwen35":
        qwen35_ref.build(d, np.random.default_rng(3), dict(qwen35_ref.TEXT, vocab_size=V, eos_token_id=sp["<|im_end|>"]))
        pack = os.path.join(d, "tiny.moe")
        subprocess.run([sys.executable, os.path.join(HERE, "convert.py"), "--model", d, "--out", pack],
                       check=True, stdout=subprocess.DEVNULL)
        return pack, os.path.join(d, "tokenizer.json")
    selftest.CFG.clear()
    selftest.CFG.update(dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=32, num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64,
                             rms_norm_eps=1e-6, rope_theta=10000.0, vocab_size=V, norm_topk_prob=True,
                             tie_word_embeddings=False, model_type="qwen3_moe", eos_token_id=[sp["<|im_end|>"]]))
    selftest.build(d, np.random.default_rng(3))
    pack = os.path.join(d, "tiny.moe")
    subprocess.run([sys.executable, os.path.join(HERE, "convert.py"), "--model", d, "--out", pack],
                   check=True, stdout=subprocess.DEVNULL)
    return pack, os.path.join(d, "tokenizer.json")


class Server:
    def __init__(self, pack, tok, extra=()):
        self.port = free_port()
        self.p = subprocess.Popen([ENGINE, "--pack", pack, "--tokenizer", tok, "--serve", "--host", "127.0.0.1",
                                   "--port", str(self.port), "--ram-gb", "0.6",
                                   "--max-answer", "24", "--threads", "2", "--max-context", "512", *extra],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("server did not start: " + self.p.stderr.read())

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Content-Type": "application/json", **(headers or {})}
        c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        data = r.read().decode("utf-8")
        c.close()
        return r.status, data

    def chat(self, messages, **kw):
        st, data = self.req("POST", "/v1/chat/completions", {"messages": messages, "temperature": 0, **kw})
        return st, (json.loads(data) if data.startswith("{") else data)

    def stream(self, messages, stop_after=None, **kw):
        """Returns (content, comments, events). stop_after: close after this many content chunks."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request("POST", "/v1/chat/completions",
                  body=json.dumps({"messages": messages, "temperature": 0, "stream": True, **kw}),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        content, comments, events, n = "", [], [], 0
        buf = b""
        while True:
            chunk = r.read1(4096) if hasattr(r, "read1") else r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                line = ev.decode("utf-8")
                if line.startswith(":"):
                    comments.append(line)
                elif line.startswith("data: "):
                    d = line[6:]
                    if d == "[DONE]":
                        c.close()
                        return content, comments, events
                    j = json.loads(d)
                    events.append(j)
                    for ch in j.get("choices", []):
                        if ch.get("delta", {}).get("content"):
                            content += ch["delta"]["content"]
                            n += 1
                            if stop_after and n >= stop_after:
                                c.close()
                                return content, comments, events
        c.close()
        return content, comments, events

    def close(self):
        self.p.terminate()
        self.p.wait(timeout=10)


def V_of(pack):
    sys.path.insert(0, HERE)
    import packfmt
    return packfmt.Pack(pack).ci("vocab_size")


def cli(pack, tok, text, n):
    out = subprocess.run([ENGINE, "--pack", pack, "--tokenizer", tok, "--prompt", text, "--max-new", str(n),
                          "--quiet", "--threads", "2"], capture_output=True, check=True).stdout
    return out.decode("utf-8", errors="replace")


def main():
    for arch in ("qwen3", "qwen35"):
        print(f"===== {'Qwen3-MoE' if arch == 'qwen3' else 'Qwen3.5/3.6-MoE (DeltaNet)'}")
        run(arch)
    print("SERVER TEST PASSED" if not FAILS else "SERVER TEST FAILED: " + ", ".join(FAILS))
    sys.exit(1 if FAILS else 0)


def run(arch):
    with tempfile.TemporaryDirectory() as d:
        pack, tok = build(d, arch)
        s = Server(pack, tok)
        try:
            print("1) page, models, stats")
            st, page = s.req("GET", "/")
            check("page is served", st == 200 and "llm-on-ssd" in page and "<script>" in page)
            st, m = s.req("GET", "/v1/models")
            check("/v1/models", st == 200 and json.loads(m)["data"][0]["id"] == "tiny")
            st, js = s.req("GET", "/api/stats")
            sj = json.loads(js)
            check("/api/stats", st == 200 and sj["phase"] == "idle" and "cap_gb" in sj["ram"])

            print("2) answers")
            msg = [{"role": "user", "content": "Hello, how are you? தமிழ் 😀"}]
            st, r1 = s.chat(msg, max_tokens=12)
            ans1 = r1["choices"][0]["message"]["content"]
            check("non-streaming answer", st == 200 and r1["usage"]["completion_tokens"] >= 1, str(r1)[:200])
            want = cli(pack, tok, msg[0]["content"], 12)
            want_c = want.split("<|im_end|>")[0].split("<|endoftext|>")[0].rstrip("\n").lstrip()
            check("greedy answer equals terminal mode", ans1 == want_c, f"{ans1!r} vs {want_c!r}")
            content, comments, events = s.stream(msg, max_tokens=12)
            check("streaming equals non-streaming", content == ans1, f"{content!r} vs {ans1!r}")
            check("stream sends reading progress", any(c.startswith(": reading") for c in comments))
            check("stream ends with usage", any(e.get("usage") for e in events))

            print("3) conversation reuse")
            conv = msg + [{"role": "assistant", "content": ans1}, {"role": "user", "content": "And tomorrow?"}]
            content2, comments2, _ = s.stream(conv, max_tokens=12)
            reused = [int(c.split()[4]) for c in comments2 if c.startswith(": reading")]
            check("second message reuses the conversation", reused and reused[0] > 10, str(comments2[:3]))
            s.stream([{"role": "user", "content": "Something completely different"}], max_tokens=6)
            full, comm_full, _ = s.stream(conv, max_tokens=12)   # after another chat: read again
            again, comm_again, _ = s.stream(conv, max_tokens=12)  # same chat again: nearly all reused
            r_full = [int(c.split()[3]) for c in comm_full if c.startswith(": reading")][0]
            r_again = [int(c.split()[3]) for c in comm_again if c.startswith(": reading")][0]
            check("reused conversation gives the same answer as reading it again",
                  content2 == full == again and r_again < r_full, f"{content2!r} / {full!r} / {again!r}")
            if arch == "qwen35":
                total = [int(c.split()[3]) for c in comm_again if c.startswith(": reading")][0]
                check("same message again resumes from the snapshot (reads 1 token)", total == 1, str(comm_again[:2]))

            print("4) queue, stop")
            results = {}

            def worker(k):
                results[k] = s.stream([{"role": "user", "content": f"Question number {k}"}], max_tokens=300)

            ths = [threading.Thread(target=worker, args=(k,)) for k in range(3)]
            for t in ths:
                t.start()
                time.sleep(0.05)
            st_busy = json.loads(s.req("GET", "/api/stats")[1])
            for t in ths:
                t.join()
            check("three parallel requests all finish", len(results) == 3 and all(v[2] for v in results.values()))
            check("later requests see their place in line",
                  any(any(c.startswith(": queue") for c in v[1]) for v in results.values()) or st_busy["waiting"] > 0)
            part, _, _ = s.stream([{"role": "user", "content": "Tell me a long story"}], stop_after=1, max_tokens=24)
            st, r = s.chat([{"role": "user", "content": "Still there?"}], max_tokens=4)
            check("server keeps working after a stopped answer", st == 200)

            print("5) long conversation, thinking")
            long = []
            for k in range(30):
                long += [{"role": "user", "content": f"Message {k}: " + "words " * 20},
                         {"role": "assistant", "content": "ok " * 10}]
            long.append({"role": "user", "content": "last"})
            _, comments3, ev3 = s.stream(long, max_tokens=8)
            check("old messages are dropped when the context is full", any(c.startswith(": dropped") for c in comments3))
            st, r = s.chat([{"role": "user", "content": "Think about it"}], max_tokens=6, think=True)
            check("thinking mode answers", st == 200)
            if arch == "qwen35":
                st, r = s.chat([{"role": "user", "content": "  padded message  "}], max_tokens=4)
                st2, r2 = s.chat([{"role": "user", "content": "padded message"}], max_tokens=4)
                check("message text is trimmed like Qwen3.5's template",
                      r["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"])
            st, r = s.req("POST", "/v1/chat/completions", {"messages": []})
            check("empty request is refused", st == 400)
            st, js = s.req("GET", "/api/stats")
            sj = json.loads(js)
            check("stats count answers", sj["jobs_done"] >= 8 and sj["last"]["gen_tok_s"] > 0, js[:200])
        finally:
            s.close()

        print("6) perplexity check")
        text = open(os.path.join(HERE, "..", "..", "trace", "corpus.txt"), encoding="utf-8").read()[:3000]
        tf = os.path.join(d, "t.txt")
        open(tf, "w", encoding="utf-8").write(text)
        out = subprocess.run([ENGINE, "--pack", pack, "--tokenizer", tok, "--perplexity", tf, "--max-tokens", "300",
                              "--threads", "2"], capture_output=True, text=True, check=True).stdout
        ppl = float(out.split()[1])
        ids = subprocess.run([ENGINE, "--tokenizer", tok, "--tokenize", tf], capture_output=True, text=True,
                             check=True).stdout.strip().split(",")[:300]
        lg = os.path.join(d, "lg.bin")
        subprocess.run([ENGINE, "--pack", pack, "--prompt-ids", ",".join(ids), "--max-new", "1", "--dump-logits", lg,
                        "--threads", "2", "--quiet"], capture_output=True, check=True)
        L = np.fromfile(lg, np.float32).reshape(-1, selftest.CFG["vocab_size"] if arch == "qwen3" else V_of(pack))[:len(ids) - 1]
        L = L.astype(np.float64)
        lse = L.max(1) + np.log(np.exp(L - L.max(1, keepdims=True)).sum(1))
        want = float(np.exp(np.mean(lse - L[np.arange(len(ids) - 1), [int(x) for x in ids[1:]]])))
        check("perplexity matches a separate calculation from the logits", abs(ppl - want) / want < 1e-4, f"{ppl} vs {want}")

        print("7) password")
        s = Server(pack, tok, ["--password", "secret"])
        try:
            st, _ = s.req("GET", "/v1/models")
            check("refused without the password", st == 401)
            st, _ = s.req("GET", "/v1/models", headers={"Authorization": "Bearer secret"})
            check("accepted with the password", st == 200)
            st, _ = s.req("GET", "/")
            check("page loads without the password (it asks for it)", st == 200)
        finally:
            s.close()


if __name__ == "__main__":
    main()
