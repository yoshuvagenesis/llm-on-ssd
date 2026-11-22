//! Expert store: reads expert blocks from the SSD with a pool of reader threads into a
//! fixed arena of slots in RAM (the cache), with prefetching.
//!
//! - The cache is one large region split into equal slots, one expert block per slot.
//!   On Linux it is backed by huge pages (2 MB) when the kernel allows it: streaming the
//!   weights from RAM is then much faster (measured +25-45%), memory is never freshly
//!   allocated per read, and RAM use is fixed and known in advance.
//! - Reads use O_DIRECT on Linux (F_NOCACHE on macOS), so the OS page cache does not
//!   silently use RAM beyond the cap. Falls back to normal reads if unsupported.
//! - Two queues: blocks needed now ("demand") are always read before guesses for later
//!   layers ("speculative"). A guess whose layer has already passed is dropped without
//!   touching the SSD. At most `max_guess_reads` guesses are read at once.
//! - A guessed block is protected from eviction until its layer runs, so prefetching works
//!   even with a tiny cache. Eviction order: wrong guesses, then least recently used.
//! - Blocks are spread over identical copies of the pack (other drives) by expert index.

use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub const ALIGN: usize = 4096;
const HUGE: usize = 2 << 20;

/// Small heap buffer aligned to 4096 bytes (used for the O_DIRECT probe).
struct ProbeBuf {
    ptr: *mut u8,
    layout: Layout,
}
impl ProbeBuf {
    fn new() -> ProbeBuf {
        let layout = Layout::from_size_align(ALIGN, ALIGN).unwrap();
        ProbeBuf { ptr: unsafe { alloc(layout) }, layout }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, ALIGN) }
    }
}
impl Drop for ProbeBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

/// One expert block in the arena. Valid while the store lives; the slot is only reused
/// after the cache entry is evicted, which never happens while a Block is held.
pub struct Block {
    ptr: usize,
    len: usize,
    slot: usize,
}
impl Block {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

struct Arena {
    base: usize,
    layout: Layout,
    slot_len: usize,
    nslots: usize,
    huge: bool,
}

impl Arena {
    fn new(slot_len: usize, nslots: usize) -> Arena {
        let total = (slot_len * nslots).div_ceil(HUGE) * HUGE;
        let layout = Layout::from_size_align(total, HUGE).unwrap();
        // Safety: non-zero size, valid alignment; memory is written by reads before use.
        let base = unsafe { alloc(layout) };
        assert!(!base.is_null(), "out of memory allocating the {} MB expert cache", total >> 20);
        let huge = advise_huge(base, total);
        Arena { base: base as usize, layout, slot_len, nslots, huge }
    }
    fn slot_ptr(&self, s: usize) -> *mut u8 {
        (self.base + s * self.slot_len) as *mut u8
    }
}

/// A 2 MB-aligned region with huge pages when possible (for --bench-kernels).
/// Returns (address, huge pages enabled). Lives until the process exits.
pub fn bench_region(len: usize) -> (usize, bool) {
    let a = Arena::new(len, 1);
    let r = (a.base, a.huge);
    std::mem::forget(a);
    r
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { dealloc(self.base as *mut u8, self.layout) }
    }
}

#[cfg(target_os = "linux")]
fn advise_huge(p: *mut u8, len: usize) -> bool {
    extern "C" {
        fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }
    const MADV_HUGEPAGE: i32 = 14;
    let mode = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").unwrap_or_default();
    if mode.contains("[never]") {
        return false;
    }
    // Safety: p..p+len is a valid allocation owned by the arena.
    unsafe { madvise(p, len, MADV_HUGEPAGE) == 0 }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge(_p: *mut u8, _len: usize) -> bool {
    false
}

enum Slot {
    Queued,         // waiting in a read queue
    Reading, // a reader is filling an arena slot
    Ready(Arc<Block>),
}

struct Entry {
    slot: Slot,
    last_use: u64,
    protect_until: u64, // layer step until which a guessed block is wanted / kept
    used: bool,
    demanded: bool, // someone needs it now (never dropped)
    guessed: bool,  // was first requested as a guess
}

struct State {
    map: HashMap<usize, Entry>,
    demand: VecDeque<usize>,
    spec: VecDeque<usize>,
    spec_inflight: usize,
    free: Vec<usize>, // free arena slots
    quit: bool,
}

#[derive(Default)]
pub struct Stats {
    pub reuse: AtomicU64,         // block was in the cache from an earlier use
    pub hits: AtomicU64,          // block was guessed and had already arrived
    pub waits: AtomicU64,         // block was queued or being read when needed
    pub misses: AtomicU64,        // block was not requested at all before it was needed
    pub unpredicted: AtomicU64,   // needed, and neither cached nor guessed when the layer routed
    pub demanded: AtomicU64,      // expert blocks needed (one per expert per layer)
    pub cancelled: AtomicU64,     // guesses withdrawn by a better prediction before reading
    pub reads: AtomicU64,         // SSD reads issued
    pub bytes_read: AtomicU64,    // bytes read from SSD
    pub read_ns: AtomicU64,       // total time inside read calls (summed over threads)
    pub prefetched: AtomicU64,    // guesses queued
    pub prefetch_used: AtomicU64, // guessed blocks that were later used
    pub dropped: AtomicU64,       // guesses dropped before reading (too late)
    pub evictions: AtomicU64,
    pub forced: AtomicU64, // evictions of still-protected guesses (cache too small)
}

pub struct ExpertStore {
    blocks: Vec<(u64, usize)>, // absolute file offset, length
    arena: Arena,
    max_spec: usize,
    state: Mutex<State>,
    ready: Condvar, // a block became ready
    work: Condvar,  // a read was queued or a slot may have become free
    step: AtomicU64,
    clock: AtomicU64, // access counter for exact least-recently-used order
    pub stats: Stats,
    pub direct: bool,
}

#[cfg(target_os = "linux")]
fn open_direct(path: &str) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    #[cfg(target_arch = "x86_64")]
    const O_DIRECT: i32 = 0o40000;
    #[cfg(target_arch = "aarch64")]
    const O_DIRECT: i32 = 0o200000;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const O_DIRECT: i32 = 0;
    OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path)
}

#[cfg(target_os = "macos")]
fn open_direct(path: &str) -> io::Result<File> {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    const F_NOCACHE: i32 = 48;
    let f = OpenOptions::new().read(true).open(path)?;
    // Safety: valid fd; F_NOCACHE takes an int argument.
    if unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(f)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_direct(path: &str) -> io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

impl ExpertStore {
    /// `paths`: the pack plus optional identical copies on other drives.
    /// `blocks`: absolute file offset and length of every expert block (all the same size).
    /// The cache gets max(cap_bytes / block, min_slots) slots.
    pub fn new(
        paths: &[String],
        blocks: Vec<(u64, usize)>,
        cap_bytes: usize,
        min_slots: usize,
        io_threads: usize,
        max_guess_reads: usize,
        try_direct: bool,
    ) -> io::Result<Arc<Self>> {
        let slot_len = blocks.iter().map(|b| b.1).max().unwrap_or(ALIGN).div_ceil(ALIGN) * ALIGN;
        let size0 = std::fs::metadata(&paths[0])?.len();
        let mut files = Vec::new();
        let mut direct = try_direct;
        for p in paths {
            if std::fs::metadata(p)?.len() != size0 {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{p} is not an identical copy of {}", paths[0])));
            }
            let mut f = None;
            if try_direct {
                if let Ok(fd) = open_direct(p) {
                    // probe one read: O_DIRECT can open fine but fail on read (e.g. tmpfs, overlayfs)
                    let mut probe = ProbeBuf::new();
                    if fd.read_exact_at(probe.as_mut_slice(), blocks[0].0).is_ok() {
                        f = Some(fd);
                    }
                }
            }
            if f.is_none() {
                direct = false;
            }
            files.push(match f {
                Some(fd) => fd,
                None => File::open(p)?,
            });
        }
        let nslots = (cap_bytes / slot_len).max(min_slots).max(1);
        let arena = Arena::new(slot_len, nslots);
        let store = Arc::new(ExpertStore {
            blocks,
            arena,
            max_spec: max_guess_reads.max(1),
            state: Mutex::new(State {
                map: HashMap::new(),
                demand: VecDeque::new(),
                spec: VecDeque::new(),
                spec_inflight: 0,
                free: (0..nslots).rev().collect(),
                quit: false,
            }),
            ready: Condvar::new(),
            work: Condvar::new(),
            step: AtomicU64::new(1),
            clock: AtomicU64::new(1),
            stats: Stats::default(),
            direct,
        });
        let files = Arc::new(files);
        for _ in 0..io_threads.max(1) {
            let files = Arc::clone(&files);
            let st = Arc::downgrade(&store);
            std::thread::spawn(move || loop {
                let Some(store) = st.upgrade() else { return };
                let Some((key, guess, slot)) = store.next_read() else { return };
                let (off, len) = store.blocks[key];
                // Safety: the slot is owned by this read until finish() publishes it.
                let dst = unsafe { std::slice::from_raw_parts_mut(store.arena.slot_ptr(slot), len) };
                let t0 = Instant::now();
                let file = &files[key % files.len()]; // spread blocks over the copies
                if let Err(e) = file.read_exact_at(dst, off) {
                    eprintln!("fatal: reading expert block {key} at {off}: {e}");
                    std::process::exit(2);
                }
                store.stats.read_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                store.stats.reads.fetch_add(1, Ordering::Relaxed);
                store.stats.bytes_read.fetch_add(len as u64, Ordering::Relaxed);
                store.finish(key, slot, len, guess);
            });
        }
        Ok(store)
    }

    pub fn cache_bytes(&self) -> usize {
        self.arena.nslots * self.arena.slot_len
    }
    pub fn huge_pages(&self) -> bool {
        self.arena.huge
    }

    /// Free slot, evicting if needed: wrong guesses first, then least recently used.
    /// As a last resort a still-protected guess is evicted (cache smaller than needed).
    fn acquire_slot(&self, st: &mut State, now: u64) -> Option<usize> {
        if let Some(s) = st.free.pop() {
            return Some(s);
        }
        let evictable = |e: &Entry, strict: bool| match &e.slot {
            Slot::Ready(b) => Arc::strong_count(b) == 1 && !e.demanded && (!strict || e.protect_until <= now),
            _ => false,
        };
        let mut victim = st.map.iter().filter(|(_, e)| evictable(e, true)).min_by_key(|(_, e)| (e.used as u8, e.last_use)).map(|(k, _)| *k);
        if victim.is_none() {
            victim = st.map.iter().filter(|(_, e)| evictable(e, false)).min_by_key(|(_, e)| (e.used as u8, e.protect_until)).map(|(k, _)| *k);
            if victim.is_some() {
                self.stats.forced.fetch_add(1, Ordering::Relaxed);
            }
        }
        let k = victim?;
        match st.map.remove(&k) {
            Some(Entry { slot: Slot::Ready(b), .. }) => {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                Some(b.slot)
            }
            _ => None,
        }
    }

    /// Next block to read: demand queue first; guesses only while still useful and while
    /// fewer than max_spec guesses are in flight. Returns (key, is_guess, slot).
    fn next_read(&self) -> Option<(usize, bool, usize)> {
        let mut st = self.state.lock().unwrap();
        loop {
            if st.quit {
                return None;
            }
            let now = self.step.load(Ordering::Relaxed);
            let from_demand = !st.demand.is_empty();
            let key = if from_demand {
                st.demand.pop_front()
            } else if st.spec_inflight < self.max_spec {
                st.spec.pop_front()
            } else {
                None
            };
            let Some(k) = key else {
                st = self.work.wait(st).unwrap();
                continue;
            };
            let Some(e) = st.map.get(&k) else { continue };
            if !matches!(e.slot, Slot::Queued) {
                continue; // already being read or ready (duplicate queue entry)
            }
            if !e.demanded && e.protect_until < now {
                st.map.remove(&k); // its layer has passed: skip the SSD read
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let guess = !e.demanded;
            let Some(slot) = self.acquire_slot(&mut st, now) else {
                // every slot is in use right now: retry shortly
                if from_demand {
                    st.demand.push_front(k);
                } else {
                    st.spec.push_front(k);
                }
                st = self.work.wait_timeout(st, Duration::from_millis(1)).unwrap().0;
                continue;
            };
            st.map.get_mut(&k).unwrap().slot = Slot::Reading;
            if guess {
                st.spec_inflight += 1;
            }
            return Some((k, guess, slot));
        }
    }

    /// Advance the layer step counter (called once per layer).
    pub fn tick(&self) -> u64 {
        self.step.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Ask for a block ahead of time. `demand` = needed by the current layer (read first,
    /// never dropped). Guesses stay wanted for `protect` layer steps.
    pub fn prefetch(&self, key: usize, protect: u64, demand: bool) {
        let now = self.step.load(Ordering::Relaxed);
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut st = self.state.lock().unwrap();
        if demand {
            self.stats.demanded.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(e) = st.map.get_mut(&key) {
            if !e.used {
                e.protect_until = e.protect_until.max(now + protect);
            }
            if demand && !e.demanded {
                e.demanded = true;
                if matches!(e.slot, Slot::Queued) {
                    st.demand.push_back(key); // promote a queued guess
                    drop(st);
                    self.work.notify_one();
                }
            }
            return;
        }
        st.map.insert(
            key,
            Entry { slot: Slot::Queued, last_use: tick, protect_until: now + protect, used: false, demanded: demand, guessed: !demand },
        );
        if demand {
            self.stats.unpredicted.fetch_add(1, Ordering::Relaxed);
            st.demand.push_back(key);
        } else {
            st.spec.push_back(key);
            self.stats.prefetched.fetch_add(1, Ordering::Relaxed);
        }
        drop(st);
        self.work.notify_one();
    }

    /// Get a block, waiting for the SSD if needed.
    pub fn get(&self, key: usize) -> Arc<Block> {
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut st = self.state.lock().unwrap();
        let mut counted = false;
        loop {
            match st.map.get_mut(&key) {
                Some(e) => {
                    if let Slot::Ready(b) = &e.slot {
                        let b = Arc::clone(b);
                        if !counted {
                            if e.used {
                                self.stats.reuse.fetch_add(1, Ordering::Relaxed);
                            } else {
                                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        if !e.used && e.guessed {
                            self.stats.prefetch_used.fetch_add(1, Ordering::Relaxed);
                        }
                        e.used = true;
                        e.protect_until = 0;
                        e.demanded = false;
                        e.last_use = now;
                        return b;
                    }
                    if !counted {
                        self.stats.waits.fetch_add(1, Ordering::Relaxed);
                        counted = true;
                    }
                    if !e.demanded {
                        e.demanded = true;
                        if matches!(e.slot, Slot::Queued) {
                            st.demand.push_back(key);
                            self.work.notify_one();
                        }
                    }
                    st = self.ready.wait(st).unwrap();
                }
                None => {
                    self.stats.misses.fetch_add(1, Ordering::Relaxed);
                    counted = true;
                    st.map.insert(
                        key,
                        Entry { slot: Slot::Queued, last_use: now, protect_until: 0, used: false, demanded: true, guessed: false },
                    );
                    self.stats.unpredicted.fetch_add(1, Ordering::Relaxed);
                    self.stats.demanded.fetch_add(1, Ordering::Relaxed);
                    st.demand.push_back(key);
                    self.work.notify_one();
                    st = self.ready.wait(st).unwrap();
                }
            }
        }
    }

    fn finish(&self, key: usize, slot: usize, len: usize, guess: bool) {
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut st = self.state.lock().unwrap();
        if guess {
            st.spec_inflight -= 1;
        }
        let block = Arc::new(Block { ptr: self.arena.slot_ptr(slot) as usize, len, slot });
        match st.map.get_mut(&key) {
            Some(e) => e.slot = Slot::Ready(block),
            None => {
                st.map.insert(
                    key,
                    Entry { slot: Slot::Ready(block), last_use: now, protect_until: 0, used: false, demanded: false, guessed: false },
                );
            }
        }
        drop(st);
        self.ready.notify_all();
        self.work.notify_one();
    }

    /// Withdraw a guess that a better prediction no longer supports. Only guesses that are
    /// still waiting in the queue are removed (nothing being read is interrupted).
    pub fn cancel_guess(&self, key: usize) {
        let mut st = self.state.lock().unwrap();
        if let Some(e) = st.map.get(&key) {
            if matches!(e.slot, Slot::Queued) && e.guessed && !e.demanded && !e.used {
                st.map.remove(&key);
                self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Mark cached blocks as just used (keys in order of increasing priority), so they are
    /// kept when the cache needs room.
    pub fn touch(&self, keys: &[usize]) {
        let mut st = self.state.lock().unwrap();
        for &k in keys {
            let t = self.clock.fetch_add(1, Ordering::Relaxed);
            if let Some(e) = st.map.get_mut(&k) {
                if matches!(e.slot, Slot::Ready(_)) {
                    e.last_use = t;
                    e.used = true;
                }
            }
        }
    }

    /// True if the block is in RAM right now.
    pub fn is_ready(&self, key: usize) -> bool {
        matches!(self.state.lock().unwrap().map.get(&key), Some(Entry { slot: Slot::Ready(_), .. }))
    }

    /// Bytes of the cache currently holding blocks (or being filled).
    pub fn cached_bytes(&self) -> usize {
        let st = self.state.lock().unwrap();
        (self.arena.nslots - st.free.len()) * self.arena.slot_len
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        if let Ok(mut st) = self.state.lock() {
            st.quit = true;
        }
        self.work.notify_all();
    }
}
