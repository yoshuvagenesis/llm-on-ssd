//! Heat protection: read CPU and NVMe temperatures from Linux sysfs and pause briefly
//! between tokens if either gets too hot, until it has cooled a few degrees.
//! Without readable sensors (e.g. macOS) the guard does nothing.

use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Latest readings, shared with the server's statistics (temperatures in 0.1 C).
#[derive(Default)]
pub struct Live {
    pub cpu_dc: AtomicU32,
    pub ssd_dc: AtomicU32,
    pub pausing: AtomicBool,
}

pub struct Guard {
    cpu: Vec<String>,
    ssd: Vec<String>,
    pub max_cpu: f32,
    pub max_ssd: f32,
    pub peak_cpu: f32,
    pub peak_ssd: f32,
    pub paused: f64,
    last: Instant,
    pub live: Arc<Live>,
}

fn read_c(path: &str) -> Option<f32> {
    fs::read_to_string(path).ok()?.trim().parse::<f32>().ok().map(|m| m / 1000.0)
}

impl Guard {
    pub fn new(max_cpu: f32, max_ssd: f32) -> Guard {
        // CPU: prefer coretemp (the CPU's own sensor), then CPU-type thermal zones;
        // board sensors such as acpitz only as a last resort (they barely react to load)
        let (mut core, mut zones, mut board, mut ssd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        if let Ok(rd) = fs::read_dir("/sys/class/hwmon") {
            for e in rd.flatten() {
                let p = e.path();
                let name = fs::read_to_string(p.join("name")).unwrap_or_default();
                match name.trim() {
                    "nvme" => ssd.push(p.join("temp1_input").to_string_lossy().into_owned()),
                    "coretemp" | "k10temp" | "zenpower" => core.push(p.join("temp1_input").to_string_lossy().into_owned()),
                    _ => {}
                }
            }
        }
        if let Ok(rd) = fs::read_dir("/sys/class/thermal") {
            for e in rd.flatten() {
                let p = e.path();
                let ty = fs::read_to_string(p.join("type")).unwrap_or_default();
                let ty = ty.trim().to_string();
                let t = p.join("temp").to_string_lossy().into_owned();
                if ty.contains("pkg") || ty.contains("cpu") || ty.contains("core") {
                    zones.push(t);
                } else if ty == "acpitz" {
                    board.push(t);
                }
            }
        }
        let cpu = if !core.is_empty() {
            core
        } else if !zones.is_empty() {
            zones
        } else {
            board
        };
        Guard { cpu, ssd, max_cpu, max_ssd, peak_cpu: 0.0, peak_ssd: 0.0, paused: 0.0, last: Instant::now(), live: Arc::new(Live::default()) }
    }

    pub fn sensors(&self) -> String {
        format!("{} CPU, {} SSD sensors", self.cpu.len(), self.ssd.len())
    }

    /// A function that reads the current (CPU, SSD) temperatures, for other threads.
    pub fn reader(&self) -> impl Fn() -> (f32, f32) + Send + 'static {
        let (cpu, ssd) = (self.cpu.clone(), self.ssd.clone());
        move || {
            let mx = |v: &[String]| v.iter().filter_map(|p| read_c(p)).fold(0f32, f32::max);
            (mx(&cpu), mx(&ssd))
        }
    }

    fn temps(&self) -> (f32, f32) {
        let mx = |v: &[String]| v.iter().filter_map(|p| read_c(p)).fold(0f32, f32::max);
        (mx(&self.cpu), mx(&self.ssd))
    }

    /// Call between tokens. Reads sensors at most every 0.5 s; pauses while too hot.
    pub fn check(&mut self) {
        if self.last.elapsed() < Duration::from_millis(500) {
            return;
        }
        self.last = Instant::now();
        let (mut c, mut s) = self.temps();
        self.live.cpu_dc.store((c * 10.0) as u32, Ordering::Relaxed);
        self.live.ssd_dc.store((s * 10.0) as u32, Ordering::Relaxed);
        self.peak_cpu = self.peak_cpu.max(c);
        self.peak_ssd = self.peak_ssd.max(s);
        let hot = |c: f32, s: f32, margin: f32| {
            (self.max_cpu > 0.0 && c >= self.max_cpu - margin) || (self.max_ssd > 0.0 && s >= self.max_ssd - margin)
        };
        if !hot(c, s, 0.0) {
            return;
        }
        let t0 = Instant::now();
        self.live.pausing.store(true, Ordering::Relaxed);
        while hot(c, s, 5.0) && t0.elapsed() < Duration::from_secs(120) {
            std::thread::sleep(Duration::from_millis(250));
            (c, s) = self.temps();
            self.live.cpu_dc.store((c * 10.0) as u32, Ordering::Relaxed);
            self.live.ssd_dc.store((s * 10.0) as u32, Ordering::Relaxed);
        }
        self.live.pausing.store(false, Ordering::Relaxed);
        self.paused += t0.elapsed().as_secs_f64();
    }
}
