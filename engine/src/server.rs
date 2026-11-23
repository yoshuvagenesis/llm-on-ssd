//! Chat server: a built-in web page plus an OpenAI-compatible API, on one port.
//!
//! - One engine thread runs the model; requests wait in a queue (first come, first served).
//! - Each request is rendered with Qwen3's chat format and compared with the tokens already
//!   in the KV cache: the common start is kept, only the rest is read.
//! - Streaming uses server-sent events in OpenAI's format. Queue position and prompt
//!   reading progress are sent as SSE comments (": queue ...", ": reading ..."), which
//!   other clients ignore and the built-in page shows.
//! - GET /api/stats reports queue, progress, speed, temperatures, RAM and cache use.

use crate::json::{self, esc, J};
use crate::model::Model;
use crate::store::ExpertStore;
use crate::thermal::{Guard, Live};
use crate::tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const PAGE: &str = include_str!("web_index.html");

pub struct Opts {
    pub host: String,
    pub port: u16,
    pub password: Option<String>,
    pub max_ctx: usize,
    pub prefill_batch: usize,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub ram_json: String, // RAM plan, already JSON
    pub qwen35: bool,     // Qwen3.5/3.6 sampling defaults
}

// ------------------------------------------------------------------ jobs and status

struct Job {
    id: u64,
    msgs: Vec<(String, String)>,
    think: bool,
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: usize,
    presence: f32,
    seed: u64,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<Ev>,
}

enum Ev {
    Reading { done: usize, total: usize, reused: usize },
    Reasoning(String),
    Content(String),
    Done { finish: &'static str, prompt: usize, completion: usize, dropped: usize },
    Error(String),
}

#[derive(Default, Clone)]
struct Status {
    phase: String, // idle | reading | generating
    prompt_total: usize,
    prompt_done: usize,
    reused: usize,
    generated: usize,
    tok_s: f64,
    read_tok_s: f64,
    started: Option<Instant>,
    jobs_done: u64,
    avg_job_s: f64,
    last_gen_tok_s: f64,
    last_read_tok_s: f64,
    hit_pct: f64,
    mb_per_tok: f64,
}

struct Shared {
    queue: Mutex<VecDeque<Job>>,
    cv: Condvar,
    busy: Mutex<Option<u64>>, // id of the job being run
    status: Mutex<Status>,
    next_id: AtomicU64,
    opts: Opts,
    live: Arc<Live>,
    store: Arc<ExpertStore>,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ------------------------------------------------------------------ sampling

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample(logits: &[f32], temp: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temp <= 0.0 || top_k == 1 {
        return logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    }
    let k = if top_k == 0 { logits.len() } else { top_k.min(logits.len()) };
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let m = logits[idx[0]];
    let mut p: Vec<f64> = idx.iter().map(|&i| (((logits[i] - m) / temp) as f64).exp()).collect();
    let s: f64 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= s);
    let mut keep = p.len();
    let mut acc = 0.0;
    for (i, v) in p.iter().enumerate() {
        acc += v;
        if acc >= top_p as f64 {
            keep = i + 1;
            break;
        }
    }
    let s: f64 = p[..keep].iter().sum();
    let mut r = rng.unit() * s;
    for i in 0..keep {
        r -= p[i];
        if r <= 0.0 {
            return idx[i] as u32;
        }
    }
    idx[keep - 1] as u32
}

// ------------------------------------------------------------------ engine thread

pub struct Engine {
    pub m: Model,
    pub guard: Guard,
    pub vocab: Vec<Vec<u8>>,
    pub tok: Tokenizer,
}

struct Ctx {
    ids: Vec<u32>,                    // tokens whose keys/values are in the KV cache
    answers: VecDeque<(String, Vec<u32>)>, // recent answers: text -> the exact tokens generated
}

struct Special {
    im_start: u32,
    im_end: u32,
    think: u32,
    end_think: u32,
}

fn render(e: &Engine, sp: &Special, ctx: &Ctx, msgs: &[(String, String)], think: bool) -> Vec<u32> {
    let t = &e.tok;
    let nl = t.encode("\n", false);
    let empty_think = t.encode("<think>\n\n</think>\n\n", true);
    // Qwen3.5/3.6's template trims message text and opens "<think>\n" itself when thinking
    let q35 = e.m.cfg.think_open;
    let mut ids = Vec::new();
    for (role, content) in msgs {
        let content = if q35 { content.trim() } else { content.as_str() };
        ids.push(sp.im_start);
        ids.extend(t.encode(&format!("{role}\n"), false));
        if role == "assistant" {
            // the model saw an empty thinking block before its answer (thinking off), so
            // history repeats it; the exact generated tokens are reused when known
            ids.extend(&empty_think);
            match ctx.answers.iter().find(|(s, _)| s.as_str() == content) {
                Some((_, a)) => ids.extend(a),
                None => ids.extend(t.encode(content, false)),
            }
        } else {
            ids.extend(t.encode(content, false));
        }
        ids.push(sp.im_end);
        ids.extend(&nl);
    }
    ids.push(sp.im_start);
    ids.extend(t.encode("assistant\n", false));
    if !think {
        ids.extend(&empty_think);
    } else if q35 {
        ids.extend(t.encode("<think>\n", true));
    }
    ids
}

/// Emit the longest valid UTF-8 prefix of `buf`, keep an incomplete tail.
fn take_utf8(buf: &mut Vec<u8>) -> String {
    let n = match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        Err(e) => e.valid_up_to() + e.error_len().unwrap(),
    };
    let s = String::from_utf8_lossy(&buf[..n]).into_owned();
    buf.drain(..n);
    s
}

pub fn run(mut e: Engine, opts: Opts) {
    let sp = Special {
        im_start: e.tok.added_id("<|im_start|>").expect("tokenizer lacks <|im_start|>"),
        im_end: e.tok.added_id("<|im_end|>").expect("tokenizer lacks <|im_end|>"),
        think: e.tok.added_id("<think>").unwrap_or(u32::MAX),
        end_think: e.tok.added_id("</think>").unwrap_or(u32::MAX),
    };
    let sh = Arc::new(Shared {
        queue: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        busy: Mutex::new(None),
        status: Mutex::new(Status { phase: "idle".into(), avg_job_s: 60.0, ..Default::default() }),
        next_id: AtomicU64::new(1),
        live: e.guard.live.clone(),
        store: e.m.store.clone(),
        opts,
    });
    {
        // keep the temperatures in the statistics fresh, also during long prompt reads
        let (read, live) = (e.guard.reader(), sh.live.clone());
        std::thread::spawn(move || loop {
            let (c, s) = read();
            live.cpu_dc.store((c * 10.0) as u32, Ordering::Relaxed);
            live.ssd_dc.store((s * 10.0) as u32, Ordering::Relaxed);
            std::thread::sleep(Duration::from_secs(1));
        });
    }
    let addr = format!("{}:{}", sh.opts.host, sh.opts.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|err| {
        eprintln!("error: cannot listen on {addr}: {err}");
        std::process::exit(1)
    });
    eprintln!("chat server ready: open http://<this machine's address>:{} in a browser", sh.opts.port);
    eprintln!("  API: http://<address>:{}/v1/chat/completions (OpenAI-compatible), model \"{}\"", sh.opts.port, sh.opts.model_name);
    {
        let sh = sh.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let sh = sh.clone();
                std::thread::spawn(move || {
                    let _ = handle(conn, &sh);
                });
            }
        });
    }
    // engine loop on this thread (the model stays on the thread that created it)
    let mut ctx = Ctx { ids: Vec::new(), answers: VecDeque::new() };
    loop {
        let job = {
            let mut q = sh.queue.lock().unwrap();
            loop {
                if let Some(j) = q.pop_front() {
                    *sh.busy.lock().unwrap() = Some(j.id);
                    break j;
                }
                q = sh.cv.wait(q).unwrap();
            }
        };
        if !job.cancel.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            run_job(&mut e, &sp, &mut ctx, &job, &sh);
            let mut st = sh.status.lock().unwrap();
            let dt = t0.elapsed().as_secs_f64();
            st.avg_job_s = if st.jobs_done == 0 { dt } else { 0.7 * st.avg_job_s + 0.3 * dt };
            st.jobs_done += 1;
            st.phase = "idle".into();
            st.started = None;
        }
        *sh.busy.lock().unwrap() = None;
    }
}

fn run_job(e: &mut Engine, sp: &Special, ctx: &mut Ctx, job: &Job, sh: &Shared) {
    let max_ctx = sh.opts.max_ctx;
    let mut msgs = job.msgs.clone();
    let mut dropped = 0;
    let mut prompt = render(e, sp, ctx, &msgs, job.think);
    // keep within the context: drop the oldest messages (not a system message, not the last)
    let need = |p: &Vec<u32>| p.len() + job.max_tokens.min(256);
    while need(&prompt) > max_ctx {
        let first = msgs.iter().position(|(r, _)| r != "system").unwrap_or(0);
        if first + 1 >= msgs.len() {
            break;
        }
        msgs.remove(first);
        dropped += 1;
        prompt = render(e, sp, ctx, &msgs, job.think);
    }
    if prompt.len() + 16 > max_ctx {
        let _ = job.tx.send(Ev::Error(format!(
            "the message is too long: {} tokens, the limit is {} (--max-context)",
            prompt.len(),
            max_ctx
        )));
        return;
    }
    let max_new = job.max_tokens.min(max_ctx - prompt.len());

    // reuse the common start of the conversation memory (models with DeltaNet layers can
    // only continue where they are or go back to their snapshot; truncate says how far)
    let mut common = ctx.ids.iter().zip(&prompt).take_while(|(a, b)| a == b).count();
    if common == prompt.len() {
        common -= 1; // need at least one token to get fresh logits
    }
    let common = e.m.truncate(common);
    ctx.ids.truncate(common);
    let total = prompt.len() - common;
    {
        let mut st = sh.status.lock().unwrap();
        *st = Status {
            phase: "reading".into(),
            prompt_total: total,
            reused: common,
            started: Some(Instant::now()),
            ..st.clone()
        };
        st.prompt_done = 0;
        st.generated = 0;
        st.tok_s = 0.0;
        st.read_tok_s = 0.0;
    }
    let _ = job.tx.send(Ev::Reading { done: 0, total, reused: common });
    let ld = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let store = e.m.store.clone();
    let s = &store.stats;
    let t_read = Instant::now();
    let mut logits = Vec::new();
    let mut done = 0;
    // with DeltaNet layers, read the prompt up to its last token, save a snapshot there
    // (so the same message can be answered again without reading everything), then the
    // last token; otherwise read it all in one go
    let recurrent = e.m.has_recurrent();
    let body_end = if recurrent { prompt.len() - 1 } else { prompt.len() };
    let mut parts: Vec<&[u32]> = prompt[common..body_end].chunks(sh.opts.prefill_batch.max(1)).collect();
    if recurrent {
        parts.push(&prompt[body_end..]);
    }
    for chunk in parts {
        if job.cancel.load(Ordering::Relaxed) {
            return;
        }
        if recurrent && e.m.pos == body_end {
            e.m.save_snapshot();
        }
        match e.m.forward_batch(chunk, false) {
            Ok(mut out) => logits = out.pop().unwrap(),
            Err(err) => {
                // the KV cache may be partly written: start the next request from scratch
                e.m.truncate(0);
                ctx.ids.clear();
                let _ = job.tx.send(Ev::Error(format!("read error: {err}")));
                return;
            }
        }
        ctx.ids.extend_from_slice(chunk);
        done += chunk.len();
        e.guard.check();
        let rt = done as f64 / t_read.elapsed().as_secs_f64().max(1e-9);
        {
            let mut st = sh.status.lock().unwrap();
            st.prompt_done = done;
            st.read_tok_s = rt;
        }
        let _ = job.tx.send(Ev::Reading { done, total, reused: common });
    }
    let read_tok_s = total as f64 / t_read.elapsed().as_secs_f64().max(1e-9);

    // generate
    sh.status.lock().unwrap().phase = "generating".into();
    let mut rng = Rng(job.seed);
    let mut gen_ids: Vec<u32> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    // Qwen3.5/3.6 open the thinking block in the prompt; Qwen3 writes "<think>" itself
    let mut in_think = job.think && e.m.cfg.think_open;
    let mut content_started = false;
    let mut content_text = String::new();
    let mut finish = "length";
    let t_gen = Instant::now();
    let (r0, b0) = (ld(&s.reuse) + ld(&s.hits) + ld(&s.waits) + ld(&s.misses), ld(&s.bytes_read));
    let reuse0 = ld(&s.reuse);
    for step in 0..max_new {
        if job.cancel.load(Ordering::Relaxed) {
            finish = "cancelled";
            break;
        }
        if job.presence != 0.0 {
            // presence penalty: every token already in this answer gets a lower score once
            for &t in &gen_ids {
                if let Some(v) = logits.get_mut(t as usize) {
                    *v -= job.presence;
                }
            }
        }
        let next = sample(&logits, job.temp, job.top_p, job.top_k, &mut rng);
        gen_ids.push(next);
        let eos = e.m.cfg.eos.contains(&next) || next == sp.im_end;
        if eos {
            finish = "stop";
        } else if next == sp.think && job.think && step == 0 {
            in_think = true;
        } else if next == sp.end_think && in_think {
            in_think = false;
            let rest = take_utf8(&mut buf);
            if !rest.is_empty() {
                let _ = job.tx.send(Ev::Reasoning(rest));
            }
        } else if let Some(b) = e.vocab.get(next as usize) {
            buf.extend_from_slice(b);
            let mut text = take_utf8(&mut buf);
            if !in_think && !content_started {
                text = text.trim_start().to_string(); // blank lines after </think>
                content_started = !text.is_empty();
            }
            if !text.is_empty() {
                if in_think {
                    let _ = job.tx.send(Ev::Reasoning(text));
                } else {
                    content_text.push_str(&text);
                    let _ = job.tx.send(Ev::Content(text));
                }
            }
        }
        {
            let mut st = sh.status.lock().unwrap();
            st.generated = step + 1;
            st.tok_s = (step + 1) as f64 / t_gen.elapsed().as_secs_f64().max(1e-9);
        }
        if eos || step + 1 == max_new {
            break; // the last token needs no forward pass
        }
        e.guard.check();
        match e.m.forward(next) {
            Ok(l) => logits = l,
            Err(err) => {
                e.m.truncate(0);
                ctx.ids.clear();
                let _ = job.tx.send(Ev::Error(format!("read error: {err}")));
                return;
            }
        }
        ctx.ids.push(next);
    }
    // bytes left over at the end (an incomplete character): send them as the terminal
    // would show them, rather than dropping them
    if !buf.is_empty() {
        let rest = String::from_utf8_lossy(&buf).into_owned();
        if in_think {
            let _ = job.tx.send(Ev::Reasoning(rest));
        } else {
            content_text.push_str(&rest);
            let _ = job.tx.send(Ev::Content(rest));
        }
    }
    // remember the exact tokens of this answer (thinking off), so the next message that
    // repeats it in the history reuses the KV cache
    if !job.think && !content_text.is_empty() {
        let mut a = gen_ids.clone();
        if a.last().is_some_and(|t| e.m.cfg.eos.contains(t) || *t == sp.im_end) {
            a.pop();
        }
        ctx.answers.push_front((content_text.clone(), a));
        ctx.answers.truncate(16);
    }
    let gen_n = gen_ids.len();
    let gen_tok_s = gen_n as f64 / t_gen.elapsed().as_secs_f64().max(1e-9);
    {
        let used = (ld(&s.reuse) + ld(&s.hits) + ld(&s.waits) + ld(&s.misses)).saturating_sub(r0).max(1);
        let mut st = sh.status.lock().unwrap();
        st.last_gen_tok_s = gen_tok_s;
        st.last_read_tok_s = if total > 1 { read_tok_s } else { st.last_read_tok_s };
        st.hit_pct = (ld(&s.reuse) - reuse0) as f64 / used as f64 * 100.0;
        st.mb_per_tok = (ld(&s.bytes_read) - b0) as f64 / 1e6 / gen_n.max(1) as f64;
    }
    let _ = job.tx.send(Ev::Done { finish, prompt: prompt.len(), completion: gen_n, dropped });
}

// ------------------------------------------------------------------ HTTP

struct Req {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn read_req(stream: &mut TcpStream) -> std::io::Result<Req> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut r = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut headers = Vec::new();
    let mut total = 0;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h)?;
        total += n;
        if n == 0 || h == "\r\n" || h == "\n" || total > 65536 {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let len: usize = headers.iter().find(|(k, _)| k == "content-length").and_then(|(_, v)| v.parse().ok()).unwrap_or(0);
    if len > 8 << 20 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "body too large"));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok(Req { method, path, headers, body })
}

fn respond(s: &mut TcpStream, code: &str, ctype: &str, body: &str) -> std::io::Result<()> {
    write!(
        s,
        "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    s.flush()
}

fn err_json(msg: &str) -> String {
    format!("{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\"}}}}", esc(msg))
}

fn handle(mut s: TcpStream, sh: &Arc<Shared>) -> std::io::Result<()> {
    let req = read_req(&mut s)?;
    let path = req.path.split('?').next().unwrap_or("/").to_string();
    if req.method == "OPTIONS" {
        return respond(&mut s, "204 No Content", "text/plain", "");
    }
    if req.method == "GET" && (path == "/" || path == "/index.html") {
        return respond(&mut s, "200 OK", "text/html; charset=utf-8", PAGE);
    }
    if req.method == "GET" && path == "/favicon.ico" {
        return respond(&mut s, "204 No Content", "image/x-icon", "");
    }
    if let Some(pw) = &sh.opts.password {
        let auth = req.headers.iter().find(|(k, _)| k == "authorization").map(|(_, v)| v.as_str()).unwrap_or("");
        let given = auth.strip_prefix("Bearer ").unwrap_or("");
        let ok = given.len() == pw.len() && given.bytes().zip(pw.bytes()).fold(0u8, |a, (x, y)| a | (x ^ y)) == 0;
        if !ok {
            return respond(&mut s, "401 Unauthorized", "application/json", &err_json("password required"));
        }
    }
    match (req.method.as_str(), path.as_str()) {
        ("GET", "/api/stats") => respond(&mut s, "200 OK", "application/json", &stats_json(sh)),
        ("GET", "/v1/models") => {
            let b = format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"created\":{},\"owned_by\":\"llm-on-ssd\"}}]}}",
                esc(&sh.opts.model_name),
                now_secs()
            );
            respond(&mut s, "200 OK", "application/json", &b)
        }
        ("POST", "/v1/chat/completions") => chat(s, sh, &req.body),
        _ => respond(&mut s, "404 Not Found", "application/json", &err_json("not found")),
    }
}

fn content_of(v: &J) -> String {
    match v {
        J::Str(s) => s.clone(),
        J::Arr(parts) => parts.iter().filter_map(|p| p.get("text").and_then(J::as_str)).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn chat(mut s: TcpStream, sh: &Arc<Shared>, body: &[u8]) -> std::io::Result<()> {
    let req = match std::str::from_utf8(body).map_err(|e| e.to_string()).and_then(json::parse) {
        Ok(j) => j,
        Err(e) => return respond(&mut s, "400 Bad Request", "application/json", &err_json(&format!("bad JSON: {e}"))),
    };
    let mut msgs = Vec::new();
    for m in req.get("messages").and_then(J::as_arr).unwrap_or(&[]) {
        let role = m.get("role").and_then(J::as_str).unwrap_or("user");
        let role = match role {
            "system" | "developer" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        msgs.push((role.to_string(), m.get("content").map(content_of).unwrap_or_default()));
    }
    if msgs.is_empty() {
        return respond(&mut s, "400 Bad Request", "application/json", &err_json("messages is empty"));
    }
    let think = req
        .get("think")
        .and_then(J::as_bool)
        .or_else(|| req.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).and_then(J::as_bool))
        .unwrap_or(false);
    let num = |k: &str| req.get(k).and_then(J::as_f64);
    // the model makers' recommended sampling (top-k 20 for both):
    //   Qwen3:   thinking 0.6 / top-p 0.95, otherwise 0.7 / 0.8
    //   Qwen3.6: thinking 1.0 / 0.95, otherwise 0.7 / 0.8; presence penalty 1.5
    let q35 = sh.opts.qwen35;
    let temp = num("temperature").unwrap_or(match (think, q35) {
        (true, true) => 1.0,
        (true, false) => 0.6,
        _ => 0.7,
    }) as f32;
    let top_p = num("top_p").unwrap_or(if think { 0.95 } else { 0.8 }) as f32;
    let top_k = num("top_k").unwrap_or(20.0).max(0.0) as usize;
    let presence = num("presence_penalty").unwrap_or(if q35 { 1.5 } else { 0.0 }) as f32;
    // thinking needs room: by default it may use the whole remaining context
    let max_tokens = num("max_completion_tokens")
        .or_else(|| num("max_tokens"))
        .map(|v| v.max(1.0) as usize)
        .unwrap_or(if think { sh.opts.max_ctx } else { sh.opts.default_max_tokens });
    let seed = num("seed").map(|v| v as u64).unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });
    let stream = req.get("stream").and_then(J::as_bool).unwrap_or(false);

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let id = sh.next_id.fetch_add(1, Ordering::Relaxed);
    {
        let mut q = sh.queue.lock().unwrap();
        q.push_back(Job { id, msgs, think, max_tokens, temp, top_p, top_k, presence, seed, cancel: cancel.clone(), tx });
        sh.cv.notify_all();
    }
    let cid = format!("chatcmpl-{id}");
    let created = now_secs();
    let model = esc(&sh.opts.model_name);

    if stream {
        s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n")?;
        let chunk = |delta: &str, finish: &str| {
            format!(
                "data: {{\"id\":\"{cid}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{model},\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{finish}}}]}}\n\n"
            )
        };
        let w = |s: &mut TcpStream, t: &str| -> bool {
            if s.write_all(t.as_bytes()).and_then(|_| s.flush()).is_err() {
                cancel.store(true, Ordering::Relaxed);
                false
            } else {
                true
            }
        };
        if !w(&mut s, &chunk("{\"role\":\"assistant\",\"content\":\"\"}", "null")) {
            return Ok(());
        }
        // tell a waiting client its place in line right away
        let (pos, eta) = queue_pos(sh, id);
        if pos > 0 && sh.busy.lock().unwrap().is_some() && !w(&mut s, &format!(": queue {pos} {eta:.0}\n\n")) {
            return Ok(());
        }
        loop {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let (pos, eta) = queue_pos(sh, id);
                    let line = if pos > 0 { format!(": queue {pos} {eta:.0}\n\n") } else { ": alive\n\n".into() };
                    if !w(&mut s, &line) {
                        return Ok(());
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Ok(Ev::Reading { done, total, reused }) => {
                    if !w(&mut s, &format!(": reading {done} {total} {reused}\n\n")) {
                        return Ok(());
                    }
                }
                Ok(Ev::Reasoning(t)) => {
                    if !w(&mut s, &chunk(&format!("{{\"reasoning_content\":{}}}", esc(&t)), "null")) {
                        return Ok(());
                    }
                }
                Ok(Ev::Content(t)) => {
                    if !w(&mut s, &chunk(&format!("{{\"content\":{}}}", esc(&t)), "null")) {
                        return Ok(());
                    }
                }
                Ok(Ev::Done { finish, prompt, completion, dropped }) => {
                    if dropped > 0 {
                        w(&mut s, &format!(": dropped {dropped}\n\n"));
                    }
                    let f = if finish == "cancelled" { "stop" } else { finish };
                    w(&mut s, &chunk("{}", &esc(f)));
                    w(
                        &mut s,
                        &format!(
                            "data: {{\"id\":\"{cid}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{model},\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":{completion},\"total_tokens\":{}}}}}\n\n",
                            prompt + completion
                        ),
                    );
                    w(&mut s, "data: [DONE]\n\n");
                    break;
                }
                Ok(Ev::Error(m)) => {
                    w(&mut s, &format!("data: {}\n\n", err_json(&m)));
                    w(&mut s, "data: [DONE]\n\n");
                    break;
                }
            }
        }
        return Ok(());
    }

    // not streaming: wait for the whole answer
    let (mut content, mut reasoning) = (String::new(), String::new());
    loop {
        match rx.recv() {
            Err(_) => return respond(&mut s, "500 Internal Server Error", "application/json", &err_json("engine stopped")),
            Ok(Ev::Reading { .. }) => {}
            Ok(Ev::Reasoning(t)) => reasoning.push_str(&t),
            Ok(Ev::Content(t)) => content.push_str(&t),
            Ok(Ev::Error(m)) => return respond(&mut s, "400 Bad Request", "application/json", &err_json(&m)),
            Ok(Ev::Done { finish, prompt, completion, .. }) => {
                let rc = if reasoning.is_empty() { String::new() } else { format!(",\"reasoning_content\":{}", esc(&reasoning)) };
                let b = format!(
                    "{{\"id\":\"{cid}\",\"object\":\"chat.completion\",\"created\":{created},\"model\":{model},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}{rc}}},\"finish_reason\":{}}}],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":{completion},\"total_tokens\":{}}}}}",
                    esc(&content),
                    esc(finish),
                    prompt + completion
                );
                return respond(&mut s, "200 OK", "application/json", &b);
            }
        }
    }
}

/// Place in line (1 = next) and an estimated wait in seconds; (0, 0) once running.
fn queue_pos(sh: &Shared, id: u64) -> (usize, f64) {
    let q = sh.queue.lock().unwrap();
    let Some(i) = q.iter().position(|j| j.id == id) else { return (0, 0.0) };
    drop(q);
    let st = sh.status.lock().unwrap();
    let busy = sh.busy.lock().unwrap().is_some();
    let running_left = if busy {
        let elapsed = st.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        (st.avg_job_s - elapsed).max(5.0)
    } else {
        0.0
    };
    (i + 1, running_left + i as f64 * st.avg_job_s)
}

fn stats_json(sh: &Shared) -> String {
    let st = sh.status.lock().unwrap().clone();
    let waiting = sh.queue.lock().unwrap().iter().filter(|j| !j.cancel.load(Ordering::Relaxed)).count();
    let s = &sh.store.stats;
    let ld = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64;
    let used = (ld(&s.reuse) + ld(&s.hits) + ld(&s.waits) + ld(&s.misses)).max(1.0);
    let elapsed = st.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
    let pausing = sh.live.pausing.load(Ordering::Relaxed);
    let phase = if pausing && st.phase != "idle" { "cooling" } else { st.phase.as_str() };
    format!(
        concat!(
            "{{\"phase\":{},\"waiting\":{},\"avg_job_s\":{:.1},",
            "\"job\":{{\"prompt_total\":{},\"prompt_done\":{},\"reused\":{},\"generated\":{},\"tok_s\":{:.2},\"read_tok_s\":{:.2},\"elapsed_s\":{:.1}}},",
            "\"last\":{{\"gen_tok_s\":{:.2},\"read_tok_s\":{:.2},\"cache_hit_pct\":{:.1},\"ssd_mb_per_token\":{:.0}}},",
            "\"heat\":{{\"cpu_c\":{:.1},\"ssd_c\":{:.1},\"pausing\":{}}},",
            "\"ram\":{},",
            "\"cache\":{{\"in_use_gb\":{:.2},\"hit_pct_total\":{:.1},\"ssd_read_gb_total\":{:.2}}},",
            "\"jobs_done\":{}}}"
        ),
        esc(phase),
        waiting,
        st.avg_job_s,
        st.prompt_total,
        st.prompt_done,
        st.reused,
        st.generated,
        st.tok_s,
        st.read_tok_s,
        elapsed,
        st.last_gen_tok_s,
        st.last_read_tok_s,
        st.hit_pct,
        st.mb_per_tok,
        sh.live.cpu_dc.load(Ordering::Relaxed) as f64 / 10.0,
        sh.live.ssd_dc.load(Ordering::Relaxed) as f64 / 10.0,
        pausing,
        sh.opts.ram_json,
        sh.store.cached_bytes() as f64 / 1e9,
        ld(&s.reuse) / used * 100.0,
        ld(&s.bytes_read) / 1e9,
        st.jobs_done
    )
}
