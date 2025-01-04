use anyhow::{bail, Context, Result};
use config::LIBEXECDIR;
use log::{debug, error, info, trace};
use process_data::{
    pci_slot::PciSlot, GpuIdentifier, GpuUsageStats, Niceness, NpuUsageStats, ProcessData,
};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    io::{Read, Write},
    process::{ChildStdin, ChildStdout, Command, Stdio},
    sync::{LazyLock, Mutex},
    time::Instant,
};
use strum_macros::Display;

use gtk::{
    gio::{Icon, ThemedIcon},
    glib::GString,
};

use crate::config;

use super::{
    boot_time, FiniteOr, FLATPAK_APP_PATH, FLATPAK_SPAWN, IS_FLATPAK, NUM_CPUS, TICK_RATE,
};

static COMPANION_PROCESS: LazyLock<Mutex<(ChildStdin, ChildStdout)>> = LazyLock::new(|| {
    let proxy_path = if *IS_FLATPAK {
        format!(
            "{}/libexec/resources/resources-processes",
            FLATPAK_APP_PATH.as_str()
        )
    } else {
        format!("{LIBEXECDIR}/resources-processes")
    };

    let child = if *IS_FLATPAK {
        debug!("Spawning resources-processes in Flatpak mode ({proxy_path})");
        Command::new(FLATPAK_SPAWN)
            .args([
                &format!(
                    "--env=RUST_LOG={}",
                    std::env::var("RUST_LOG=resources").unwrap_or("warn".into())
                ),
                "--host",
                proxy_path.as_str(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap()
    } else {
        debug!("Spawning resources-processes in native mode ({proxy_path})");
        Command::new(proxy_path)
            .arg(&format!(
                "--env=RUST_LOG={}",
                std::env::var("RUST_LOG=resources").unwrap_or("warn".into())
            ))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap()
    };

    let stdin = child.stdin.unwrap();
    let stdout = child.stdout.unwrap();

    Mutex::new((stdin, stdout))
});

/// Represents a process that can be found within procfs.
#[derive(Debug, Clone, PartialEq)]
pub struct Process {
    pub data: ProcessData,
    pub executable_path: String,
    pub executable_name: String,
    pub icon: Icon,
    pub cpu_time_last: u64,
    pub timestamp_last: u64,
    pub read_bytes_last: Option<u64>,
    pub write_bytes_last: Option<u64>,
    pub gpu_usage_stats_last: BTreeMap<GpuIdentifier, GpuUsageStats>,
    pub npu_usage_stats_last: BTreeMap<PciSlot, NpuUsageStats>,
    pub display_name: String,
}

// TODO: Better name?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum ProcessAction {
    TERM,
    STOP,
    KILL,
    CONT,
}

impl Process {
    /// Returns a `Vec` containing all currently running processes.
    ///
    /// # Errors
    ///
    /// Will return `Err` if there are problems traversing and
    /// parsing procfs
    pub fn all_data() -> Result<Vec<ProcessData>> {
        trace!("all_data() called");

        let start = Instant::now();
        let output = {
            trace!("Acquiring companion process lock");
            let mut process = COMPANION_PROCESS.lock().unwrap();
            trace!("Writing b\"\\n\" into companion process stdin");
            let _ = process.0.write_all(b"\n");
            trace!("Flushing");
            let _ = process.0.flush();

            let mut len_bytes = [0_u8; (usize::BITS / 8) as usize];

            trace!("Reading companion process output length as little-endian");
            process.1.read_exact(&mut len_bytes)?;

            let len = usize::from_le_bytes(len_bytes);
            trace!("Companion process output is {len} bytes long");

            let mut output_bytes = vec![0; len];
            trace!("Reading companion process output");
            process.1.read_exact(&mut output_bytes)?;

            output_bytes
        };

        let elapsed = start.elapsed();
        trace!("Companion process was done in {elapsed:.2?}");

        trace!("Parsing companion process output");
        let parsed =
            rmp_serde::from_slice(&output).context("unable to decode companion process output");

        let elapsed = start.elapsed();
        trace!("all_data() done in {elapsed:.2?}");

        parsed
    }

    pub fn from_process_data(process_data: ProcessData) -> Self {
        let executable_path = process_data
            .commandline
            .split('\0')
            .nth(0)
            .and_then(|nul_split| nul_split.split(" --").nth(0)) // chromium (and thus everything based on it) doesn't use \0 as delimiter
            .unwrap_or(&process_data.commandline)
            .to_string();

        let executable_name = executable_path
            .split('/')
            .nth_back(0)
            .unwrap_or(&process_data.commandline)
            .to_string();

        let read_bytes_last = if process_data.read_bytes.is_some() {
            Some(0)
        } else {
            None
        };

        let write_bytes_last = if process_data.write_bytes.is_some() {
            Some(0)
        } else {
            None
        };

        let display_name = if executable_name.starts_with(&process_data.comm) {
            executable_name.clone()
        } else {
            process_data.comm.clone()
        };

        Self {
            executable_path,
            executable_name,
            data: process_data,
            icon: ThemedIcon::new("generic-process").into(),
            cpu_time_last: 0,
            timestamp_last: 0,
            read_bytes_last,
            write_bytes_last,
            gpu_usage_stats_last: Default::default(),
            npu_usage_stats_last: Default::default(),
            display_name,
        }
    }

    /// Tries to run a command unprivileged and then privileged if permissions were missing
    fn maybe_pkexec_command<S: AsRef<OsStr>, I: IntoIterator<Item = S>>(
        command: S,
        args: I,
    ) -> Result<i32> {
        let args = args
            .into_iter()
            .map(|s| s.as_ref().to_os_string())
            .collect::<Vec<_>>();

        let status_code = if *IS_FLATPAK {
            debug!(
                "Executing command: {} --host {} {}",
                FLATPAK_SPAWN,
                command.as_ref().to_string_lossy(),
                args.join(&OsString::from(" ")).to_string_lossy()
            );

            Command::new(FLATPAK_SPAWN)
                .arg("--host")
                .arg(command.as_ref())
                .args(args.clone())
                .output()?
                .status
                .code()
                .context("no status code?")?
        } else {
            debug!(
                "Executing command: {} {}",
                command.as_ref().to_string_lossy(),
                args.join(&OsString::from(" ")).to_string_lossy()
            );

            Command::new(command.as_ref())
                .args(args.clone())
                .output()?
                .status
                .code()
                .context("no status code?")?
        };

        if status_code == libc::EPERM || status_code == libc::EACCES {
            let pkexec_status_code = if *IS_FLATPAK {
                debug!(
                    "Received EPERM, executing command: {} --host pkexec --disable-internal-agent {} {}", 
                    FLATPAK_SPAWN,
                    command.as_ref().to_string_lossy(),
                    args.join(&OsString::from(" ")).to_string_lossy()
                );
                Command::new(FLATPAK_SPAWN)
                    .args(["--host", "pkexec", "--disable-internal-agent"])
                    .arg(command)
                    .args(args)
                    .output()?
                    .status
                    .code()
                    .context("no status code?")?
            } else {
                debug!(
                    "Received EPERM or EACCES, executing command: pkexec --disable-internal-agent {} {}", 
                    command.as_ref().to_string_lossy(),
                    args.join(&OsString::from(" ")).to_string_lossy()
                );
                Command::new("pkexec")
                    .arg("--disable-internal-agent")
                    .arg(command)
                    .args(args)
                    .output()?
                    .status
                    .code()
                    .context("no status code?")?
            };

            Ok(pkexec_status_code)
        } else {
            Ok(status_code)
        }
    }

    pub fn adjust<I: IntoIterator<Item = bool>>(
        &self,
        niceness: Niceness,
        affinity: I,
    ) -> Result<()> {
        let adjust_path = if *IS_FLATPAK {
            format!(
                "{}/libexec/resources/resources-adjust",
                FLATPAK_APP_PATH.as_str()
            )
        } else {
            format!("{LIBEXECDIR}/resources-adjust")
        };

        let affinity_string = affinity
            .into_iter()
            .map(|b| if b { '1' } else { '0' })
            .collect::<String>();

        debug!("Trying to adjust with niceness = {niceness} and affinity = {affinity_string}");

        let result = Self::maybe_pkexec_command(
            adjust_path,
            [
                self.data.pid.to_string(),
                niceness.to_string(),
                affinity_string,
            ],
        );

        if let Ok(return_code) = result {
            if return_code == 0 {
                info!("Successfully adjusted {}", self.data.pid);
                Ok(())
            } else {
                bail!("non-zero return code: {return_code}")
            }
        } else if let Err(err) = result {
            Err(err)
        } else {
            bail!("unknown error")
        }
    }

    pub fn execute_process_action(&self, action: ProcessAction) -> Result<()> {
        let action_string = action.to_string();

        let kill_path = if *IS_FLATPAK {
            format!(
                "{}/libexec/resources/resources-kill",
                FLATPAK_APP_PATH.as_str()
            )
        } else {
            format!("{LIBEXECDIR}/resources-kill")
        };

        let result = Self::maybe_pkexec_command(
            kill_path,
            [self.data.pid.to_string(), action_string.clone()],
        );

        if let Ok(return_code) = result {
            if return_code == 0 || return_code == 3 {
                info!("Successfully {action_string}ed {}", self.data.pid);
                Ok(())
            } else {
                error!(
                    "Couldn't {action_string} {}, return code: {return_code}",
                    self.data.pid
                );
                bail!("non-zero return code: {return_code}")
            }
        } else if let Err(err) = result {
            error!(
                "Unknown error while trying to {action_string} {}\n{err}\n{}",
                self.data.pid,
                err.backtrace()
            );
            Err(err)
        } else {
            bail!("unknown error")
        }
    }

    #[must_use]
    pub fn cpu_time_ratio(&self) -> f32 {
        if self.cpu_time_last == 0 {
            0.0
        } else {
            let delta_cpu_time = (self
                .data
                .user_cpu_time
                .saturating_add(self.data.system_cpu_time))
            .saturating_sub(self.cpu_time_last) as f32
                * 1000.0;
            let delta_time = self.data.timestamp.saturating_sub(self.timestamp_last);

            (delta_cpu_time
                / (delta_time
                    .saturating_mul(*TICK_RATE as u64)
                    .saturating_mul(*NUM_CPUS as u64)) as f32)
                .finite_or_default()
        }
    }

    #[must_use]
    pub fn read_speed(&self) -> Option<f64> {
        if let (Some(read_bytes), Some(read_bytes_last)) =
            (self.data.read_bytes, self.read_bytes_last)
        {
            if self.timestamp_last == 0 {
                Some(0.0)
            } else {
                let bytes_delta = read_bytes.saturating_sub(read_bytes_last) as f64;
                let time_delta = self.data.timestamp.saturating_sub(self.timestamp_last) as f64;
                Some((bytes_delta / time_delta) * 1000.0)
            }
        } else {
            None
        }
    }

    #[must_use]
    pub fn write_speed(&self) -> Option<f64> {
        if let (Some(write_bytes), Some(write_bytes_last)) =
            (self.data.write_bytes, self.write_bytes_last)
        {
            if self.timestamp_last == 0 {
                Some(0.0)
            } else {
                let bytes_delta = write_bytes.saturating_sub(write_bytes_last) as f64;
                let time_delta = self.data.timestamp.saturating_sub(self.timestamp_last) as f64;
                Some((bytes_delta / time_delta) * 1000.0)
            }
        } else {
            None
        }
    }

    #[must_use]
    pub fn gpu_usage(&self) -> f32 {
        let mut returned_gpu_usage = 0.0;
        for (gpu, usage) in &self.data.gpu_usage_stats {
            if let Some(old_usage) = self.gpu_usage_stats_last.get(gpu) {
                let this_gpu_usage = if usage.nvidia {
                    usage.gfx as f32 / 100.0
                } else if old_usage.gfx == 0 {
                    0.0
                } else {
                    ((usage.gfx.saturating_sub(old_usage.gfx) as f32)
                        / (self.data.timestamp.saturating_sub(self.timestamp_last) as f32)
                            .finite_or_default())
                        / 1_000_000.0
                };

                if this_gpu_usage > returned_gpu_usage {
                    returned_gpu_usage = this_gpu_usage;
                }
            }
        }

        returned_gpu_usage
    }

    #[must_use]
    pub fn enc_usage(&self) -> f32 {
        let mut returned_gpu_usage = 0.0;
        for (gpu, usage) in &self.data.gpu_usage_stats {
            if let Some(old_usage) = self.gpu_usage_stats_last.get(gpu) {
                let this_gpu_usage = if usage.nvidia {
                    usage.enc as f32 / 100.0
                } else if old_usage.enc == 0 {
                    0.0
                } else {
                    ((usage.enc.saturating_sub(old_usage.enc) as f32)
                        / (self.data.timestamp.saturating_sub(self.timestamp_last) as f32)
                            .finite_or_default())
                        / 1_000_000.0
                };

                if this_gpu_usage > returned_gpu_usage {
                    returned_gpu_usage = this_gpu_usage;
                }
            }
        }

        returned_gpu_usage
    }

    #[must_use]
    pub fn dec_usage(&self) -> f32 {
        let mut returned_gpu_usage = 0.0;
        for (gpu, usage) in &self.data.gpu_usage_stats {
            if let Some(old_usage) = self.gpu_usage_stats_last.get(gpu) {
                let this_gpu_usage = if usage.nvidia {
                    usage.dec as f32 / 100.0
                } else if old_usage.dec == 0 {
                    0.0
                } else {
                    ((usage.dec.saturating_sub(old_usage.dec) as f32)
                        / (self.data.timestamp.saturating_sub(self.timestamp_last) as f32)
                            .finite_or_default())
                        / 1_000_000.0
                };

                if this_gpu_usage > returned_gpu_usage {
                    returned_gpu_usage = this_gpu_usage;
                }
            }
        }

        returned_gpu_usage
    }

    #[must_use]
    pub fn gpu_mem_usage(&self) -> u64 {
        self.data
            .gpu_usage_stats
            .values()
            .map(|stats| stats.mem)
            .sum()
    }

    #[must_use]
    pub fn npu_usage(&self) -> f32 {
        let mut returned_npu_usage = 0.0;
        for (npu, usage) in &self.data.npu_usage_stats {
            if let Some(old_usage) = self.npu_usage_stats_last.get(npu) {
                let this_npu_usage = if old_usage.usage == 0 {
                    0.0
                } else {
                    ((usage.usage.saturating_sub(old_usage.usage) as f32)
                        / (self.data.timestamp.saturating_sub(self.timestamp_last) as f32)
                            .finite_or_default())
                        / 1_000_000.0
                };

                if this_npu_usage > returned_npu_usage {
                    returned_npu_usage = this_npu_usage;
                }
            }
        }

        returned_npu_usage
    }

    #[must_use]
    pub fn npu_mem_usage(&self) -> u64 {
        self.data
            .npu_usage_stats
            .values()
            .map(|stats| stats.mem)
            .sum()
    }

    #[must_use]
    pub fn starttime(&self) -> f64 {
        self.data.starttime as f64 / *TICK_RATE as f64
    }

    pub fn running_since(&self) -> Result<GString> {
        boot_time()
            .and_then(|boot_time| {
                boot_time
                    .add_seconds(self.starttime())
                    .context("unable to add seconds to boot time")
            })
            .and_then(|time| time.format("%c").context("unable to format running_since"))
    }

    pub fn sanitize_cmdline<S: AsRef<str>>(cmdline: S) -> Option<String> {
        let cmdline = cmdline.as_ref();
        if cmdline.is_empty() {
            None
        } else {
            Some(cmdline.replace('\0', " "))
        }
    }
}
