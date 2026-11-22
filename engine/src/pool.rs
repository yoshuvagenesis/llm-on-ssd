//! A small persistent thread pool (std only). `run(n, f)` calls f(0..n) spread over the
//! workers and the calling thread, and returns when all calls are finished, so `f` may
//! borrow local data. Replaces starting new threads for every operation.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type TaskFn<'a> = dyn Fn(usize) + Sync + 'a;

#[derive(Clone, Copy)]
struct FnPtr(*const TaskFn<'static>);
unsafe impl Send for FnPtr {}

struct Job {
    gen: u64,
    f: Option<FnPtr>,
    n: usize,
    quit: bool,
}

struct Shared {
    job: Mutex<Job>,
    start: Condvar,
    finished: Condvar,
    next: AtomicUsize,
    done: AtomicUsize,
    busy: AtomicUsize, // workers currently inside a job
}

pub struct Pool {
    sh: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    pub threads: usize,
}

impl Pool {
    /// `threads` = total threads doing work, including the caller.
    pub fn new(threads: usize) -> Pool {
        let threads = threads.max(1);
        let sh = Arc::new(Shared {
            job: Mutex::new(Job { gen: 0, f: None, n: 0, quit: false }),
            start: Condvar::new(),
            finished: Condvar::new(),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            busy: AtomicUsize::new(0),
        });
        let workers = (1..threads)
            .map(|_| {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let (f, n) = {
                            let mut j = sh.job.lock().unwrap();
                            while j.gen == seen && !j.quit {
                                j = sh.start.wait(j).unwrap();
                            }
                            if j.quit {
                                return;
                            }
                            seen = j.gen;
                            match j.f {
                                // job still open: join it (run() waits for us via `busy`)
                                Some(f) => {
                                    sh.busy.fetch_add(1, Ordering::SeqCst);
                                    (f, j.n)
                                }
                                // woke up after the caller already finished every task
                                None => continue,
                            }
                        };
                        loop {
                            let i = sh.next.fetch_add(1, Ordering::SeqCst);
                            if i >= n {
                                break;
                            }
                            // Safety: run() keeps the closure alive until busy == 0 and done == n.
                            unsafe { (*f.0)(i) };
                            sh.done.fetch_add(1, Ordering::SeqCst);
                        }
                        let _g = sh.job.lock().unwrap();
                        sh.busy.fetch_sub(1, Ordering::SeqCst);
                        sh.finished.notify_all();
                    }
                })
            })
            .collect();
        Pool { sh, workers, threads }
    }

    pub fn run<F: Fn(usize) + Sync>(&self, n: usize, f: F) {
        if n == 0 {
            return;
        }
        if self.threads == 1 || n == 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let fref: &TaskFn<'_> = &f;
        // Safety: the lifetime is erased only for the duration of this call; we do not
        // return before every worker has left the job (busy == 0).
        let fptr: *const TaskFn<'static> = unsafe { std::mem::transmute::<&TaskFn<'_>, &TaskFn<'static>>(fref) };
        let sh = &self.sh;
        {
            let mut j = sh.job.lock().unwrap();
            while sh.busy.load(Ordering::SeqCst) != 0 {
                j = sh.finished.wait(j).unwrap();
            }
            sh.next.store(0, Ordering::SeqCst);
            sh.done.store(0, Ordering::SeqCst);
            j.f = Some(FnPtr(fptr));
            j.n = n;
            j.gen += 1;
        }
        sh.start.notify_all();
        loop {
            let i = sh.next.fetch_add(1, Ordering::SeqCst);
            if i >= n {
                break;
            }
            f(i);
            sh.done.fetch_add(1, Ordering::SeqCst);
        }
        let mut j = sh.job.lock().unwrap();
        while sh.done.load(Ordering::SeqCst) < n || sh.busy.load(Ordering::SeqCst) != 0 {
            j = sh.finished.wait(j).unwrap();
        }
        j.f = None;
    }

    /// Split `out` into about `threads` chunks (multiples of `align`) and fill them in parallel:
    /// f(index of first element, chunk).
    pub fn chunks<F: Fn(usize, &mut [f32]) + Sync>(&self, out: &mut [f32], align: usize, f: F) {
        let n = out.len();
        if self.threads <= 1 || n <= align {
            f(0, out);
            return;
        }
        let chunk = n.div_ceil(self.threads).div_ceil(align) * align;
        let nchunks = n.div_ceil(chunk);
        let base = out.as_mut_ptr() as usize;
        self.run(nchunks, |i| {
            let s = i * chunk;
            let e = (s + chunk).min(n);
            // Safety: chunks are disjoint and inside `out`, which outlives run().
            let sl = unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(s), e - s) };
            f(s, sl);
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.job.lock().unwrap().quit = true;
        self.sh.start.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
