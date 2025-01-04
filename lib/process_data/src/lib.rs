pub mod pci_slot;

use anyhow::{bail, Context, Result};
use glob::glob;
use lazy_regex::{lazy_regex, Lazy, Regex};
use log::{debug, trace, warn};
use nutype::nutype;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::struct_wrappers::device::{ProcessInfo, ProcessUtilizationSample};
use nvml_wrapper::{Device, Nvml};
use pci_slot::PciSlot;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Display;
use std::os::linux::fs::MetadataExt;
use std::path::Path;
use std::str::FromStr;
use std::sync::{LazyLock, RwLock};
use std::time::SystemTime;

const STAT_OFFSET: usize = 2; // we split the stat contents where the executable name ends, which is the second element
const STAT_PARENT_PID: usize = 3 - STAT_OFFSET;
const STAT_USER_CPU_TIME: usize = 13 - STAT_OFFSET;
const STAT_SYSTEM_CPU_TIME: usize = 14 - STAT_OFFSET;
const STAT_NICE: usize = 18 - STAT_OFFSET;
const STAT_STARTTIME: usize = 21 - STAT_OFFSET;

const GPU_DRIVER_NAMES: &[&str] = &["amdgpu", "i915"];
const NPU_DRIVER_NAMES: &[&str] = &["amdxdna_accel_driver"];

const MAJOR: u32 = 226;

static USERS_CACHE: LazyLock<HashMap<libc::uid_t, String>> = LazyLock::new(|| unsafe {
    debug!("Initializing users cache…");
    let users: HashMap<libc::uid_t, String> = uzers::all_users()
        .map(|user| {
            trace!("Found user {}", user.name().to_string_lossy());
            (user.uid(), user.name().to_string_lossy().to_string())
        })
        .collect();
    debug!("Found {} users", users.len());
    users
});

static PAGESIZE: LazyLock<usize> = LazyLock::new(sysconf::pagesize);

static NUM_CPUS: LazyLock<usize> = LazyLock::new(num_cpus::get);

static RE_UID: Lazy<Regex> = lazy_regex!(r"Uid:\s*(\d+)");

static RE_AFFINITY: Lazy<Regex> = lazy_regex!(r"Cpus_allowed:\s*([0-9A-Fa-f]+)");

static RE_SWAP_USAGGE: Lazy<Regex> = lazy_regex!(r"VmSwap:\s*([0-9]+)\s*kB");

static RE_IO_READ: Lazy<Regex> = lazy_regex!(r"read_bytes:\s*(\d+)");

static RE_IO_WRITE: Lazy<Regex> = lazy_regex!(r"write_bytes:\s*(\d+)");

static RE_DRM_DRIVER: Lazy<Regex> = lazy_regex!(r"drm-driver:\s*(.+)");

static RE_DRM_PDEV: Lazy<Regex> =
    lazy_regex!(r"drm-pdev:\s*([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])");

// AMD only
static RE_DRM_ENGINE_NPU_AMDXDNA: Lazy<Regex> =
    lazy_regex!(r"drm-engine-npu-amdxdna:\s*(\d+)\s*ns");

// AMD only
static RE_DRM_ENGINE_GFX: Lazy<Regex> = lazy_regex!(r"drm-engine-gfx:\s*(\d+)\s*ns");

// AMD only
static RE_DRM_ENGINE_COMPUTE: Lazy<Regex> = lazy_regex!(r"drm-engine-compute:\s*(\d+)\s*ns");

// AMD only
static RE_DRM_ENGINE_ENC: Lazy<Regex> = lazy_regex!(r"drm-engine-enc:\s*(\d+)\s*ns");

// AMD only
static RE_DRM_ENGINE_DEC: Lazy<Regex> = lazy_regex!(r"drm-engine-dec:\s*(\d+)\s*ns");

// AMD only
static RE_DRM_MEMORY_VRAM: Lazy<Regex> = lazy_regex!(r"drm-memory-vram:\s*(\d+)\s*KiB");

// AMD only
static RE_DRM_MEMORY_GTT: Lazy<Regex> = lazy_regex!(r"drm-memory-gtt:\s*(\d+)\s*KiB");

// Intel and v3d only
static RE_DRM_ENGINE_RENDER: Lazy<Regex> = lazy_regex!(r"drm-engine-render:\s*(\d+)\s*ns");

// Intel only
static RE_DRM_ENGINE_VIDEO: Lazy<Regex> = lazy_regex!(r"drm-engine-video:\s*(\d+)\s*ns");

static RE_DRM_TOTAL_MEMORY: Lazy<Regex> = lazy_regex!(r"drm-total-memory:\s*(\d+)\s*KiB");

static NVML: Lazy<Result<Nvml, NvmlError>> = Lazy::new(|| {
    debug!("Initializing connection to NVML…");
    Nvml::init().inspect_err(|err| warn!("Unable to connect to NVML: {err}"))
});

static NVML_DEVICES: Lazy<Vec<(PciSlot, Device)>> = Lazy::new(|| {
    if let Ok(nvml) = NVML.as_ref() {
        debug!("Looking for NVIDIA devices…");
        let device_count = nvml.device_count().unwrap_or(0);
        let mut return_vec = Vec::with_capacity(device_count as usize);
        for i in 0..device_count {
            if let Ok(gpu) = nvml.device_by_index(i) {
                if let Ok(pci_slot) = gpu.pci_info().map(|pci_info| pci_info.bus_id) {
                    let pci_slot = PciSlot::from_str(&pci_slot).unwrap();
                    debug!(
                        "Found {} at {}",
                        gpu.name().unwrap_or("N/A".into()),
                        pci_slot
                    );
                    return_vec.push((pci_slot, gpu));
                }
            }
        }
        return_vec
    } else {
        Vec::new()
    }
});

static NVIDIA_PROCESSES_STATS: Lazy<RwLock<HashMap<PciSlot, Vec<ProcessUtilizationSample>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static NVIDIA_PROCESS_INFOS: Lazy<RwLock<HashMap<PciSlot, Vec<ProcessInfo>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[nutype(
    validate(less_or_equal = 19),
    validate(greater_or_equal = -20),
    derive(Debug, Default, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, Copy, FromStr, Deref, TryFrom, Display),
    default = 0
)]
pub struct Niceness(i8);

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub enum Containerization {
    #[default]
    None,
    Flatpak,
    Snap,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, Copy, PartialOrd, Ord)]
pub enum GpuIdentifier {
    PciSlot(PciSlot),
    Enumerator(usize),
}

impl Default for GpuIdentifier {
    fn default() -> Self {
        GpuIdentifier::Enumerator(0)
    }
}

impl Display for GpuIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuIdentifier::PciSlot(pci_slot) => write!(f, "{}", pci_slot),
            GpuIdentifier::Enumerator(e) => write!(f, "{}", e),
        }
    }
}

/// Represents GPU usage statistics per-process. Depending on the GPU manufacturer (which should be determined in
/// Resources itself), these numbers need to interpreted differently
///
/// AMD (default): gfx, enc and dec are nanoseconds spent for that process
///
/// Nvidia: Process info is gathered through NVML, thus gfx, enc and dec are percentages from 0-100 (timestamps
/// are irrelevant, nvidia bool is set to true)
///
/// Intel: enc and dec are not separated, both are accumulated in enc, also mem is always going to be 0
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub struct GpuUsageStats {
    pub gfx: u64,
    pub mem: u64,
    pub enc: u64,
    pub dec: u64,
    pub nvidia: bool,
}

/// Represents NPU usage statistics per-process.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub struct NpuUsageStats {
    pub usage: u64,
    pub mem: u64,
}

/// Data that could be transferred using `resources-processes`, separated from
/// `Process` mainly due to `Icon` not being able to derive `Serialize` and
/// `Deserialize`.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessData {
    pub pid: libc::pid_t,
    pub parent_pid: libc::pid_t,
    pub user: String,
    pub comm: String,
    pub commandline: String,
    pub user_cpu_time: u64,
    pub system_cpu_time: u64,
    pub niceness: Niceness,
    pub affinity: Vec<bool>,
    pub memory_usage: usize,
    pub swap_usage: usize,
    pub starttime: u64, // in clock ticks, see man proc(5)!
    pub cgroup: Option<String>,
    pub containerization: Containerization,
    pub read_bytes: Option<u64>,
    pub write_bytes: Option<u64>,
    pub timestamp: u64,
    /// Key: PCI Slot ID of the GPU
    pub gpu_usage_stats: BTreeMap<GpuIdentifier, GpuUsageStats>,
    pub npu_usage_stats: BTreeMap<PciSlot, NpuUsageStats>,
}

impl ProcessData {
    fn sanitize_cgroup<S: AsRef<str>>(cgroup: S) -> Option<String> {
        let cgroups_v2_line = cgroup.as_ref().split('\n').find(|s| s.starts_with("0::"))?;
        if cgroups_v2_line.ends_with(".scope") {
            let cgroups_segments: Vec<&str> = cgroups_v2_line.split('-').collect();
            if cgroups_segments.len() > 1 {
                cgroups_segments
                    .get(cgroups_segments.len() - 2)
                    .map(|s| unescape::unescape(s).unwrap_or_else(|| (*s).to_string()))
            } else {
                None
            }
        } else if cgroups_v2_line.ends_with(".service") {
            let cgroups_segments: Vec<&str> = cgroups_v2_line.split('/').collect();
            if let Some(last) = cgroups_segments.last() {
                last[0..last.len() - 8]
                    .split('@')
                    .next()
                    .map(|s| unescape::unescape(s).unwrap_or_else(|| s.to_string()))
                    .map(|s| {
                        if s.contains("dbus-:") {
                            s.split('-').last().unwrap_or(&s).to_string()
                        } else {
                            s
                        }
                    })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_uid(proc_path: &Path) -> Result<u32> {
        let status = std::fs::read_to_string(proc_path.join("status"))?;
        if let Some(captures) = RE_UID.captures(&status) {
            let first_num_str = captures.get(1).context("no uid found")?;
            first_num_str
                .as_str()
                .parse::<u32>()
                .context("couldn't parse uid in /status")
        } else {
            Ok(0)
        }
    }

    pub fn update_nvidia_stats() {
        {
            let mut stats = NVIDIA_PROCESSES_STATS.write().unwrap();
            stats.clear();
            stats.extend(Self::nvidia_process_stats());
        }
        {
            let mut infos = NVIDIA_PROCESS_INFOS.write().unwrap();
            infos.clear();
            infos.extend(Self::nvidia_process_infos());
        }
    }

    pub fn all_process_data() -> Result<Vec<Self>> {
        Self::update_nvidia_stats();

        let mut process_data = vec![];
        for entry in glob("/proc/[0-9]*/").context("unable to glob")?.flatten() {
            let data = ProcessData::try_from_path(&entry);

            if let Ok(data) = data {
                process_data.push(data);
            }
        }

        Ok(process_data)
    }

    pub fn try_from_path<P: AsRef<Path>>(proc_path: P) -> Result<Self> {
        let proc_path = proc_path.as_ref();
        let pid = proc_path
            .file_name()
            .context("proc_path terminates in ..")?
            .to_str()
            .context("can't turn OsStr to str")?
            .parse()?;

        trace!("Inspecting process {pid}…");

        trace!("Reading info files…");
        let stat = std::fs::read_to_string(proc_path.join("stat"))
            .inspect_err(|err| trace!("Error reading 'stat': {err}"))?;
        let statm = std::fs::read_to_string(proc_path.join("statm"))
            .inspect_err(|err| trace!("Error reading 'statm': {err}"))?;
        let status = std::fs::read_to_string(proc_path.join("status"))
            .inspect_err(|err| trace!("Error reading 'status': {err}"))?;
        let comm = std::fs::read_to_string(proc_path.join("comm"))
            .inspect_err(|err| trace!("Error reading 'comm': {err}"))?;
        let commandline = std::fs::read_to_string(proc_path.join("cmdline"))
            .inspect_err(|err| trace!("Error reading 'cmdline': {err}"))?;
        let io = std::fs::read_to_string(proc_path.join("io"))
            .inspect_err(|err| trace!("Error reading 'io': {err}"))
            .ok();

        let user = USERS_CACHE
            .get(&Self::get_uid(proc_path)?)
            .cloned()
            .unwrap_or(String::from("root"));

        let stat = stat
            .split(')') // since we don't care about the pid or the executable name, split after the executable name to make our life easier
            .last()
            .context("stat doesn't have ')'")
            .inspect_err(|err| trace!("Can't parse 'stat': {err}"))?
            .split(' ')
            .skip(1) // the first element would be a space, let's ignore that
            .collect::<Vec<_>>();

        let statm = statm.split(' ').collect::<Vec<_>>();

        let comm = comm.replace('\n', "");

        // -2 to accommodate for only collecting after the second item (which is the executable name as mentioned above)
        let parent_pid = stat
            .get(STAT_PARENT_PID)
            .context("wrong stat file format")
            .and_then(|x| x.parse().context("couldn't parse stat file content to int"))
            .inspect_err(|err| trace!("Can't parse parent pid from 'stat': {err}"))?;
        let user_cpu_time = stat
            .get(STAT_USER_CPU_TIME)
            .context("wrong stat file format")
            .and_then(|x| x.parse().context("couldn't parse stat file content to int"))
            .inspect_err(|err| trace!("Can't parse user cpu time from 'stat': {err}"))?;
        let system_cpu_time = stat
            .get(STAT_SYSTEM_CPU_TIME)
            .context("wrong stat file format")
            .and_then(|x| x.parse().context("couldn't parse stat file content to int"))
            .inspect_err(|err| trace!("Can't parse system cpu time from 'stat': {err}"))?;
        let nice = stat
            .get(STAT_NICE)
            .context("wrong stat file format")
            .and_then(|x| x.parse().context("couldn't parse stat file content to int"))
            .inspect_err(|err| trace!("Can't parse nice from 'stat': {err}"))?;
        let starttime = stat
            .get(STAT_STARTTIME)
            .context("wrong stat file format")
            .and_then(|x| x.parse().context("couldn't parse stat file content to int"))
            .inspect_err(|err| trace!("Can't parse start time from 'stat': {err}"))?;

        let mut affinity = Vec::with_capacity(*NUM_CPUS);
        RE_AFFINITY
            .captures(&status)
            .and_then(|captures| captures.get(1))
            .map(|capture| capture.as_str())
            .unwrap_or_default()
            .chars()
            .map(|c| c.to_digit(16).unwrap_or_default())
            .rev()
            .for_each(|int| {
                // we want the bits and there are 4 bits in a hex digit
                (0..4).for_each(|i| {
                    // this if should prevent wrong size affinity vecs if the thread count is not divisible by 4
                    if affinity.len() < *NUM_CPUS {
                        affinity.push((int & (1 << i)) != 0);
                    }
                });
            });

        let swap_usage = RE_SWAP_USAGGE
            .captures(&status)
            .and_then(|captures| captures.get(1))
            .map(|capture| capture.as_str())
            .unwrap_or_default()
            .parse::<usize>()
            .unwrap_or_default() // kworkers don't have swap usage
            .saturating_mul(1000);

        let memory_usage = statm
            .get(1)
            .context("wrong statm file format")
            .and_then(|x| {
                x.parse::<usize>()
                    .context("couldn't parse statm file content")
            })
            .inspect_err(|err| trace!("Can't parse memory usage from 'statm': {err}"))?
            .saturating_sub(
                statm
                    .get(2)
                    .context("wrong statm file format")
                    .and_then(|x| {
                        x.parse::<usize>()
                            .context("couldn't parse statm file content")
                    })?,
            )
            .saturating_mul(*PAGESIZE);

        let cgroup = std::fs::read_to_string(proc_path.join("cgroup"))
            .inspect_err(|err| trace!("Can't read cgroup: {err}"))
            .ok()
            .and_then(Self::sanitize_cgroup);

        let containerization = if commandline.starts_with("/snap/") {
            Containerization::Snap
        } else if proc_path.join("root").join(".flatpak-info").exists() {
            Containerization::Flatpak
        } else {
            Containerization::None
        };

        let read_bytes = io.as_ref().and_then(|io| {
            RE_IO_READ
                .captures(io)
                .and_then(|captures| captures.get(1))
                .and_then(|capture| capture.as_str().parse::<u64>().ok())
        });

        let write_bytes = io.as_ref().and_then(|io| {
            RE_IO_WRITE
                .captures(io)
                .and_then(|captures| captures.get(1))
                .and_then(|capture| capture.as_str().parse::<u64>().ok())
        });

        let gpu_usage_stats = Self::gpu_usage_stats(proc_path, pid);

        let npu_usage_stats = Self::npu_usage_stats(proc_path, pid).unwrap_or_default();

        let timestamp = unix_as_millis();

        Ok(Self {
            pid,
            parent_pid,
            user,
            comm,
            commandline,
            user_cpu_time,
            system_cpu_time,
            niceness: nice,
            affinity,
            memory_usage,
            swap_usage,
            starttime,
            cgroup,
            containerization,
            read_bytes,
            write_bytes,
            timestamp,
            gpu_usage_stats,
            npu_usage_stats,
        })
    }

    /// Returns the fd_num and the plausibility of whether this file might contain drm fdinfo data.
    /// This function is cautious and will signal plausibility if there's an error during evaluation.
    fn drm_fdinfo_plausible<P: AsRef<Path>>(
        fdinfo_path: P,
        pid: libc::pid_t,
        seen_fds: &HashSet<usize>,
    ) -> (bool, usize) {
        let fdinfo_path = fdinfo_path.as_ref();

        // if our fd is 0, 1 or 2 it's probably just a std stream so skip it
        let fd_num = fdinfo_path
            .file_name()
            .and_then(|osstr| osstr.to_str())
            .unwrap_or("0")
            .parse::<usize>()
            .unwrap_or(0);
        if fd_num <= 2 {
            trace!(
                "fdinfo {fd_num} deemed as not plausible. Reason: fd_num ≤ 2 (probably std stream)"
            );
            return (false, fd_num);
        }

        let _file = std::fs::File::open(&fdinfo_path);
        if _file.is_err() {
            trace!("fdinfo {fd_num} deemed as not plausible. Reason: File can't be opened");
            return (false, fd_num);
        }
        let file = _file.unwrap();

        let _metadata = file.metadata();
        if _metadata.is_err() {
            trace!(
                "fdinfo {fd_num} deemed as not plausible. Reason: File's metadata can't be read"
            );
            return (false, fd_num);
        }
        let metadata = _metadata.unwrap();

        if !metadata.is_file() {
            trace!("fdinfo {fd_num} deemed as not plausible. Reason: Not a file");
            return (false, fd_num);
        }

        // Adapted from nvtop's `is_drm_fd()`
        // https://github.com/Syllo/nvtop/blob/master/src/extract_processinfo_fdinfo.c
        let fd_path = fdinfo_path.to_str().map(|s| s.replace("fdinfo", "fd"));
        if let Some(fd_path) = fd_path {
            if let Ok(fd_metadata) = std::fs::metadata(fd_path) {
                if (fd_metadata.st_mode() & libc::S_IFMT) != libc::S_IFCHR {
                    trace!("fdinfo {fd_num} deemed as not plausible. Reason: Wrong st_mode");
                    return (false, fd_num);
                }
                let major = unsafe { libc::major(fd_metadata.st_rdev()) };
                if major != MAJOR {
                    trace!(
                        "fdinfo {fd_num} deemed as not plausible. Reason: Wrong major (expected: {MAJOR}, got: {major})"
                    );
                    return (false, fd_num);
                }
            }
        }

        // Adapted from nvtop's `processinfo_sweep_fdinfos()`
        // https://github.com/Syllo/nvtop/blob/master/src/extract_processinfo_fdinfo.c
        // if we've already seen the file this fd refers to, skip
        let not_unique = seen_fds.iter().any(|seen_fd| unsafe {
            syscalls::syscall!(syscalls::Sysno::kcmp, pid, pid, 0, fd_num, *seen_fd).unwrap_or(0)
                == 0
        });
        if not_unique {
            trace!("fdinfo {fd_num} deemed as not plausible. Reason: kcmp indicated that we've already seen this file");
            return (false, fd_num);
        }

        trace!("fdinfo {fd_num} deemed as plausible");
        (true, fd_num)
    }

    fn gpu_usage_stats(proc_path: &Path, pid: i32) -> BTreeMap<GpuIdentifier, GpuUsageStats> {
        trace!("Gathering GPU stats…");
        let nvidia_stats = Self::nvidia_gpu_stats_all(pid);
        let mut other_stats = Self::other_gpu_usage_stats(proc_path, pid).unwrap_or_default();
        other_stats.extend(nvidia_stats);
        other_stats
    }

    fn other_gpu_usage_stats(
        proc_path: &Path,
        pid: i32,
    ) -> Result<BTreeMap<GpuIdentifier, GpuUsageStats>> {
        let fdinfo_dir = proc_path.join("fdinfo");

        let mut seen_fds = HashSet::new();

        let mut return_map = BTreeMap::new();
        for entry in std::fs::read_dir(fdinfo_dir)?.flatten() {
            let fdinfo_path = entry.path();

            let (plausible, fd_num) = Self::drm_fdinfo_plausible(&fdinfo_path, pid, &seen_fds);
            if !plausible {
                continue;
            }

            seen_fds.insert(fd_num);

            if let Ok((pci_slot, stats)) = Self::read_gpu_fdinfo(&fdinfo_path) {
                return_map
                    .entry(pci_slot)
                    .and_modify(|existing_value: &mut GpuUsageStats| {
                        if stats.gfx > existing_value.gfx {
                            existing_value.gfx = stats.gfx;
                        }
                        if stats.dec > existing_value.dec {
                            existing_value.dec = stats.dec;
                        }
                        if stats.enc > existing_value.enc {
                            existing_value.enc = stats.enc;
                        }
                        if stats.mem > existing_value.mem {
                            existing_value.mem = stats.mem;
                        }
                    })
                    .or_insert(stats);
            }
        }

        Ok(return_map)
    }

    fn npu_usage_stats(proc_path: &Path, pid: i32) -> Result<BTreeMap<PciSlot, NpuUsageStats>> {
        trace!("Gathering NPU stats…");
        let fdinfo_dir = proc_path.join("fdinfo");

        let mut seen_fds = HashSet::new();

        let mut return_map = BTreeMap::new();
        for entry in std::fs::read_dir(fdinfo_dir)?.flatten() {
            let fdinfo_path = entry.path();

            let (plausible, fd_num) = Self::drm_fdinfo_plausible(&fdinfo_path, pid, &seen_fds);
            if !plausible {
                continue;
            }

            seen_fds.insert(fd_num);

            if let Ok((pci_slot, stats)) = Self::read_npu_fdinfo(&fdinfo_path) {
                return_map
                    .entry(pci_slot)
                    .and_modify(|existing_value: &mut NpuUsageStats| {
                        if stats.usage > existing_value.usage {
                            existing_value.usage = stats.usage;
                        }
                        if stats.mem > existing_value.mem {
                            existing_value.mem = stats.mem;
                        }
                    })
                    .or_insert(stats);
            }
        }

        Ok(return_map)
    }

    fn read_npu_fdinfo<P: AsRef<Path>>(fdinfo_path: P) -> Result<(PciSlot, NpuUsageStats)> {
        trace!(
            "Reading and parsing {} for NPU stats…",
            fdinfo_path.as_ref().to_string_lossy()
        );

        let content = std::fs::read_to_string(fdinfo_path.as_ref())?;

        let pci_slot = RE_DRM_PDEV
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| PciSlot::from_str(capture.as_str()).ok())
            .context("can't parse PCI slot ID")?;

        let driver = RE_DRM_DRIVER
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .map(|capture| capture.as_str())
            .unwrap_or_default();

        if !NPU_DRIVER_NAMES.contains(&driver) {
            trace!("Driver '{driver}' is not known to be NPU-related, skipping");
            bail!("this is not an NPU")
        }

        let usage = RE_DRM_ENGINE_NPU_AMDXDNA
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let total_memory = RE_DRM_TOTAL_MEMORY
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default()
            .saturating_mul(1024);

        let stats = NpuUsageStats {
            usage,
            mem: total_memory,
        };

        trace!("Success reading NPU data for {pci_slot}: {stats:?}");

        return Ok((pci_slot, stats));
    }

    fn read_gpu_fdinfo<P: AsRef<Path>>(fdinfo_path: P) -> Result<(GpuIdentifier, GpuUsageStats)> {
        trace!(
            "Reading and parsing {} for GPU stats…",
            fdinfo_path.as_ref().to_string_lossy()
        );

        let content = std::fs::read_to_string(fdinfo_path.as_ref())?;

        let driver = RE_DRM_DRIVER
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .map(|capture| capture.as_str())
            .unwrap_or_default();

        if !GPU_DRIVER_NAMES.contains(&driver) {
            trace!("Driver {driver} is not known to be GPU-related, skipping");
            bail!("this is not a GPU");
        }

        let gpu_identifier = RE_DRM_PDEV
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| PciSlot::from_str(capture.as_str()).ok())
            .map(|pci_slot| GpuIdentifier::PciSlot(pci_slot))
            .unwrap_or_default();

        let gfx = RE_DRM_ENGINE_GFX
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let render = RE_DRM_ENGINE_RENDER
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let compute = RE_DRM_ENGINE_COMPUTE
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let enc = RE_DRM_ENGINE_ENC
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let video = RE_DRM_ENGINE_VIDEO
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let dec = RE_DRM_ENGINE_DEC
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default();

        let vram = RE_DRM_MEMORY_VRAM
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default()
            .saturating_mul(1024);

        let gtt = RE_DRM_MEMORY_GTT
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default()
            .saturating_mul(1024);

        let total_memory = RE_DRM_TOTAL_MEMORY
            .captures(&content)
            .and_then(|captures| captures.get(1))
            .and_then(|capture| capture.as_str().parse::<u64>().ok())
            .unwrap_or_default()
            .saturating_mul(1024);

        let stats = GpuUsageStats {
            gfx: gfx.saturating_add(render).saturating_add(compute),
            mem: vram.saturating_add(gtt).saturating_add(total_memory),
            enc: enc.saturating_add(video),
            dec,
            nvidia: false,
        };

        return Ok((gpu_identifier, stats));
    }

    fn nvidia_gpu_stats_all(pid: i32) -> BTreeMap<GpuIdentifier, GpuUsageStats> {
        trace!("Gathering NVIDIA GPU stats…");

        let mut return_map = BTreeMap::new();

        for (pci_slot, _) in NVML_DEVICES.iter() {
            if let Ok(stats) = Self::nvidia_gpu_stats(pid, *pci_slot) {
                return_map.insert(GpuIdentifier::PciSlot(pci_slot.to_owned()), stats);
            }
        }

        return_map
    }

    fn nvidia_gpu_stats(pid: i32, pci_slot: PciSlot) -> Result<GpuUsageStats> {
        trace!("Gathering GPU stats for NVIDIA GPU at {pci_slot}…");
        let this_process_stats = NVIDIA_PROCESSES_STATS
            .read()
            .unwrap()
            .get(&pci_slot)
            .context("couldn't find GPU with this PCI slot")?
            .iter()
            .filter(|process| process.pid == pid as u32)
            .map(|stats| (stats.sm_util, stats.enc_util, stats.dec_util))
            .reduce(|acc, curr| (acc.0 + curr.0, acc.1 + curr.1, acc.2 + curr.2));

        let this_process_mem_stats: u64 = NVIDIA_PROCESS_INFOS
            .read()
            .unwrap()
            .get(&pci_slot)
            .context("couldn't find GPU with this PCI slot")?
            .iter()
            .filter(|process| process.pid == pid as u32)
            .map(|stats| match stats.used_gpu_memory {
                UsedGpuMemory::Unavailable => 0,
                UsedGpuMemory::Used(bytes) => bytes,
            })
            .sum();

        let gpu_stats = GpuUsageStats {
            gfx: this_process_stats.unwrap_or_default().0 as u64,
            mem: this_process_mem_stats,
            enc: this_process_stats.unwrap_or_default().1 as u64,
            dec: this_process_stats.unwrap_or_default().2 as u64,
            nvidia: true,
        };
        Ok(gpu_stats)
    }

    fn nvidia_process_infos() -> HashMap<PciSlot, Vec<ProcessInfo>> {
        trace!("Refreshing NVIDIA process infos…");
        let mut return_map = HashMap::new();

        for (pci_slot, gpu) in NVML_DEVICES.iter() {
            let mut comp_gfx_stats = gpu.running_graphics_processes().unwrap_or_default();
            comp_gfx_stats.extend(gpu.running_compute_processes().unwrap_or_default());

            return_map.insert(pci_slot.to_owned(), comp_gfx_stats);
        }

        return_map
    }

    fn nvidia_process_stats() -> HashMap<PciSlot, Vec<ProcessUtilizationSample>> {
        trace!("Refreshing NVIDIA process stats…");
        let mut return_map = HashMap::new();

        for (pci_slot, gpu) in NVML_DEVICES.iter() {
            return_map.insert(
                pci_slot.to_owned(),
                gpu.process_utilization_stats(
                    unix_as_millis()
                        .saturating_mul(1000)
                        .saturating_sub(5_000_000),
                )
                .unwrap_or_default(),
            );
        }

        return_map
    }
}

pub fn unix_as_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
