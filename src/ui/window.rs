use process_data::{Niceness, ProcessData};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use adw::{prelude::*, subclass::prelude::*, ToolbarView};
use adw::{Toast, ToastOverlay};
use anyhow::{Context, Result};
use gtk::glib::{clone, timeout_future, GString, MainContext};
use gtk::{gio, glib, Widget};
use log::{info, trace, warn};

use crate::application::Application;
use crate::config::PROFILE;
use crate::gui::ARGS;
use crate::i18n::{i18n, i18n_f, ni18n_f};
use crate::ui::pages::applications::ResApplications;
use crate::ui::pages::battery::ResBattery;
use crate::ui::pages::drive::ResDrive;
use crate::ui::pages::processes::ResProcesses;
use crate::utils::app::AppsContext;
use crate::utils::battery::{Battery, BatteryData};
use crate::utils::cpu::{self, CpuData};
use crate::utils::drive::{Drive, DriveData};
use crate::utils::gpu::{Gpu, GpuData};
use crate::utils::memory::MemoryData;
use crate::utils::network::{NetworkData, NetworkInterface};
use crate::utils::npu::{Npu, NpuData};
use crate::utils::process::{Process, ProcessAction};
use crate::utils::settings::SETTINGS;

use super::pages::gpu::ResGPU;
use super::pages::network::ResNetwork;
use super::pages::npu::ResNPU;
use super::pages::{applications, processes};

#[derive(Debug, Clone)]
pub enum Action {
    ManipulateProcesses(ProcessAction, Vec<libc::pid_t>, ToastOverlay),
    ManipulateApp(ProcessAction, String, ToastOverlay),
    AdjustProcess(libc::pid_t, Niceness, Vec<bool>, String, ToastOverlay),
}

mod imp {
    use std::{cell::RefCell, collections::HashMap};

    use crate::{
        config::VERSION,
        ui::{
            pages::{
                applications::ResApplications, cpu::ResCPU, memory::ResMemory,
                processes::ResProcesses,
            },
            widgets::stack_sidebar::ResStackSidebar,
        },
        utils::app::AppsContext,
    };

    use super::*;

    use async_channel::{unbounded, Receiver, Sender};
    use gtk::CompositeTemplate;
    use log::debug;
    use process_data::{pci_slot::PciSlot, GpuIdentifier};

    #[derive(Debug, CompositeTemplate)]
    #[template(resource = "/net/nokyan/Resources/ui/window.ui")]
    pub struct MainWindow {
        #[template_child]
        pub split_view: TemplateChild<adw::OverlaySplitView>,
        #[template_child]
        pub processor_window_title: TemplateChild<adw::WindowTitle>,
        #[template_child]
        pub resources_sidebar: TemplateChild<ResStackSidebar>,
        #[template_child]
        pub content_stack: TemplateChild<gtk::Stack>,
        #[template_child]
        pub cpu: TemplateChild<ResCPU>,
        #[template_child]
        pub cpu_page: TemplateChild<gtk::StackPage>,
        #[template_child]
        pub applications: TemplateChild<ResApplications>,
        #[template_child]
        pub applications_page: TemplateChild<gtk::StackPage>,
        #[template_child]
        pub processes: TemplateChild<ResProcesses>,
        #[template_child]
        pub processes_page: TemplateChild<gtk::StackPage>,
        #[template_child]
        pub memory: TemplateChild<ResMemory>,
        #[template_child]
        pub memory_page: TemplateChild<gtk::StackPage>,

        pub drive_pages: RefCell<HashMap<PathBuf, adw::ToolbarView>>,

        pub network_pages: RefCell<HashMap<PathBuf, adw::ToolbarView>>,

        pub battery_pages: RefCell<HashMap<PathBuf, adw::ToolbarView>>,

        pub gpu_pages: RefCell<HashMap<GpuIdentifier, (Gpu, adw::ToolbarView)>>,

        pub npu_pages: RefCell<HashMap<PciSlot, (Npu, adw::ToolbarView)>>,

        pub apps_context: RefCell<AppsContext>,

        pub sender: Sender<Action>,
        pub receiver: RefCell<Option<Receiver<Action>>>,
    }

    impl Default for MainWindow {
        fn default() -> Self {
            let (sender, r) = unbounded();
            let receiver = RefCell::new(Some(r));

            Self {
                drive_pages: RefCell::default(),
                network_pages: RefCell::default(),
                battery_pages: RefCell::default(),
                split_view: TemplateChild::default(),
                resources_sidebar: TemplateChild::default(),
                content_stack: TemplateChild::default(),
                applications: TemplateChild::default(),
                applications_page: TemplateChild::default(),
                processes: TemplateChild::default(),
                processes_page: TemplateChild::default(),
                cpu: TemplateChild::default(),
                cpu_page: TemplateChild::default(),
                memory: TemplateChild::default(),
                memory_page: TemplateChild::default(),
                apps_context: Default::default(),
                sender,
                receiver,
                processor_window_title: TemplateChild::default(),
                gpu_pages: RefCell::default(),
                npu_pages: RefCell::default(),
            }
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for MainWindow {
        const NAME: &'static str = "MainWindow";
        type Type = super::MainWindow;
        type ParentType = adw::ApplicationWindow;

        fn class_init(klass: &mut Self::Class) {
            Self::bind_template(klass);
        }

        // You must call `Widget`'s `init_template()` within `instance_init()`.
        fn instance_init(obj: &glib::subclass::InitializingObject<Self>) {
            obj.init_template();
        }
    }

    impl ObjectImpl for MainWindow {
        fn constructed(&self) {
            self.parent_constructed();
            let obj = self.obj();

            // Devel Profile
            if PROFILE == "Devel" {
                obj.add_css_class("devel");
                obj.set_title(Some(
                    &format!("{} ({})", obj.title().unwrap_or_default(), VERSION).trim(),
                ));
            }

            // Load latest window state
            obj.load_window_size();
        }
    }

    impl WidgetImpl for MainWindow {}

    impl WindowImpl for MainWindow {
        // Save window state on delete event
        fn close_request(&self) -> glib::Propagation {
            debug!("Closing the application…");

            if let Err(err) = self.obj().save_window_size() {
                warn!("Failed to save window state, {}", &err);
            }

            // Pass close request on to the parent
            self.parent_close_request()
        }
    }

    impl ApplicationWindowImpl for MainWindow {}

    impl AdwApplicationWindowImpl for MainWindow {}
}

glib::wrapper! {
    pub struct MainWindow(ObjectSubclass<imp::MainWindow>)
        @extends gtk::Widget, gtk::Window, gtk::ApplicationWindow, adw::ApplicationWindow,
        @implements gio::ActionMap, gio::ActionGroup, gtk::Root;
}

struct RefreshData {
    cpu_data: Option<CpuData>,
    mem_data: Option<Result<MemoryData>>,
    gpu_data: Vec<GpuData>,
    npu_data: Vec<NpuData>,
    drive_paths: Vec<PathBuf>,
    drive_data: Vec<DriveData>,
    network_paths: Vec<PathBuf>,
    network_data: Vec<NetworkData>,
    battery_paths: Vec<PathBuf>,
    battery_data: Vec<BatteryData>,
    process_data: Vec<ProcessData>,
}

impl MainWindow {
    pub fn new(app: &Application) -> Self {
        trace!("Creating MainWindow GObject…");

        let window = glib::Object::builder::<Self>()
            .property("application", app)
            .build();

        if let Some(receiver) = &*window.imp().receiver.borrow() {
            let main_context = MainContext::default();
            main_context.spawn_local(clone!(
                #[strong]
                receiver,
                #[weak]
                window,
                async move {
                    while let Ok(action) = receiver.recv().await {
                        window.process_action(action);
                    }
                }
            ));
        }
        window.setup_widgets();
        window
    }

    fn get_selected_page(&self) -> Option<Widget> {
        self.imp()
            .content_stack
            .visible_child()
            .and_downcast::<adw::ToolbarView>()
            .and_then(|toolbar| toolbar.content())
    }

    pub fn shortcut_toggle_search(&self) {
        let imp = self.imp();

        let selected_page = self.get_selected_page().unwrap();

        if selected_page.is::<ResApplications>() {
            imp.applications.toggle_search();
        } else if selected_page.is::<ResProcesses>() {
            imp.processes.toggle_search();
        }
    }

    pub fn shortcut_manipulate_app_process(&self, process_action: ProcessAction) {
        let imp = self.imp();

        let selected_page = self.get_selected_page().unwrap();

        if selected_page.is::<ResApplications>() {
            if let Some(app_item) = imp.applications.get_selected_app_entry() {
                imp.applications
                    .open_app_action_dialog(&app_item, process_action);
            }
        } else if selected_page.is::<ResProcesses>() {
            let selected = imp.processes.get_selected_process_entries();
            if !selected.is_empty() {
                imp.processes
                    .open_process_action_dialog(selected, process_action);
            }
        }
    }

    pub fn shortcut_information_app_process(&self) {
        let imp = self.imp();

        let selected_page = self.get_selected_page().unwrap();

        if selected_page.is::<ResApplications>() {
            if let Some(app_item) = imp.applications.get_selected_app_entry() {
                imp.applications.open_info_dialog(&app_item);
            }
        } else if selected_page.is::<ResProcesses>() {
            let selected = imp.processes.get_selected_process_entries();
            if selected.len() == 1 {
                imp.processes.open_info_dialog(&selected[0]);
            }
        }
    }

    pub fn shortcut_process_options(&self) {
        let imp = self.imp();

        let selected_page = self.get_selected_page().unwrap();

        if selected_page.is::<ResProcesses>() {
            let selected = imp.processes.get_selected_process_entries();
            if selected.len() == 1 {
                imp.processes.open_options_dialog(&selected[0]);
            }
        }
    }

    fn init_gpu_pages(self: &MainWindow, gpus: &[Gpu]) {
        let imp = self.imp();

        for (i, gpu) in gpus.iter().enumerate() {
            let page = ResGPU::new();

            let tab_name = if gpus.len() > 1 {
                i18n_f("GPU {}", &[&(i + 1).to_string()])
            } else {
                i18n("GPU")
            };

            page.set_tab_name(&*tab_name);

            let added_page = if let Ok(gpu_name) = gpu.name() {
                self.add_page(&page, &gpu_name, &tab_name)
            } else {
                self.add_page(&page, &tab_name, &tab_name)
            };

            page.init(gpu, i as u32);

            imp.gpu_pages
                .borrow_mut()
                .insert(gpu.gpu_identifier(), (gpu.clone(), added_page));
        }
    }

    fn init_npu_pages(self: &MainWindow) -> Vec<Npu> {
        let imp = self.imp();

        let npus = Npu::get_npus().unwrap_or_default();

        for (i, npu) in npus.iter().enumerate() {
            let page = ResNPU::new();

            let tab_name = if npus.len() > 1 {
                i18n_f("NPU {}", &[&(i + 1).to_string()])
            } else {
                i18n("NPU")
            };

            page.set_tab_name(&*tab_name);

            let added_page = if let Ok(npu_name) = npu.name() {
                self.add_page(&page, &npu_name, &tab_name)
            } else {
                self.add_page(&page, &tab_name, &tab_name)
            };

            page.init(npu, i as u32);

            imp.npu_pages
                .borrow_mut()
                .insert(npu.pci_slot(), (npu.clone(), added_page));
        }

        npus
    }

    fn setup_widgets(&self) {
        trace!("Setting up Application widgets…");
        let imp = self.imp();

        let gpus = Gpu::get_gpus().unwrap_or_default();

        if !ARGS.disable_gpu_monitoring {
            self.init_gpu_pages(&gpus);
        }

        imp.resources_sidebar.set_stack(&imp.content_stack);

        if SETTINGS.show_search_on_start() {
            // we want the search bar to show up for both but also let the last viewed page grab the focus, so order is
            // important here
            if SETTINGS.last_viewed_page() == applications::TAB_ID {
                imp.processes.toggle_search();
                imp.applications.toggle_search();
            } else if SETTINGS.last_viewed_page() == processes::TAB_ID {
                imp.applications.toggle_search();
                imp.processes.toggle_search();
            }
        }

        if ARGS.disable_process_monitoring {
            self.remove_page(imp.applications_page.child().downcast_ref().unwrap());
            self.remove_page(imp.processes_page.child().downcast_ref().unwrap());
        } else {
            *imp.apps_context.borrow_mut() = AppsContext::new(
                gpus.iter()
                    .filter(|gpu| gpu.combined_media_engine().unwrap_or_default())
                    .map(Gpu::gpu_identifier)
                    .collect(),
            );
            imp.applications.init(imp.sender.clone());
            imp.processes.init(imp.sender.clone());
        }

        if ARGS.disable_cpu_monitoring {
            self.remove_page(imp.cpu_page.child().downcast_ref().unwrap());
        } else {
            let cpu_info = cpu::CpuInfo::get()
                .context("unable to get CPUInfo")
                .unwrap();
            if let Some(model_name) = cpu_info.model_name.as_deref() {
                imp.processor_window_title.set_title(model_name);
                imp.processor_window_title.set_subtitle(&i18n("Processor"));
            }
            imp.cpu.init(cpu_info);
        }

        if ARGS.disable_memory_monitoring {
            self.remove_page(imp.memory_page.child().downcast_ref().unwrap());
        } else {
            imp.memory.init();
        }

        if !ARGS.disable_npu_monitoring {
            self.init_npu_pages();
        }

        let main_context = MainContext::default();

        main_context.spawn_local(clone!(
            #[weak(rename_to = this)]
            self,
            async move {
                this.periodic_refresh_all().await;
            }
        ));
    }

    fn gather_refresh_data(logical_cpus: usize, gpus: &[Gpu], npus: &[Npu]) -> RefreshData {
        let start = Instant::now();

        trace!("Gathering refresh data of all devices…");

        let cpu_data = if ARGS.disable_cpu_monitoring {
            None
        } else {
            Some(CpuData::new(logical_cpus))
        };

        let mem_data = if ARGS.disable_memory_monitoring {
            None
        } else {
            Some(MemoryData::new())
        };

        let mut gpu_data = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let data = GpuData::new(gpu);

            gpu_data.push(data);
        }

        let mut npu_data = Vec::with_capacity(npus.len());
        for npu in npus {
            let data = NpuData::new(npu);

            npu_data.push(data);
        }

        let drive_paths = if ARGS.disable_drive_monitoring {
            Vec::new()
        } else {
            Drive::get_sysfs_paths().unwrap_or_default()
        };
        let mut drive_data = Vec::with_capacity(drive_paths.len());
        for path in &drive_paths {
            drive_data.push(DriveData::new(path));
        }

        let network_paths = if ARGS.disable_network_interface_monitoring {
            Vec::new()
        } else {
            NetworkInterface::get_sysfs_paths().unwrap_or_default()
        };
        let mut network_data = Vec::with_capacity(network_paths.len());
        for path in &network_paths {
            network_data.push(NetworkData::new(path));
        }

        let battery_paths = if ARGS.disable_battery_monitoring {
            Vec::new()
        } else {
            Battery::get_sysfs_paths().unwrap_or_default()
        };
        let mut battery_data = Vec::with_capacity(battery_paths.len());
        for path in &battery_paths {
            battery_data.push(BatteryData::new(path));
        }

        let process_data = if ARGS.disable_process_monitoring {
            Vec::new()
        } else {
            Process::all_data()
                .inspect_err(|e| {
                    warn!(
                        "Unable to update process and app data! Is resources-processes running?\n{e}\n{}",
                        e.backtrace()
                    );
                })
                .unwrap_or_default()
        };

        let refresh_data = RefreshData {
            cpu_data,
            mem_data,
            gpu_data,
            npu_data,
            drive_paths,
            drive_data,
            network_paths,
            network_data,
            battery_paths,
            battery_data,
            process_data,
        };

        trace!("Finished gathering refresh data in {:.2?}", start.elapsed());

        refresh_data
    }

    fn refresh_ui(&self, refresh_data: RefreshData) {
        let start = Instant::now();

        trace!("Refreshing UI using gathered data…");

        let imp = self.imp();

        let RefreshData {
            cpu_data,
            mem_data,
            gpu_data,
            npu_data,
            drive_paths,
            drive_data,
            network_paths,
            network_data,
            battery_paths,
            battery_data,
            process_data,
        } = refresh_data;

        /*
         * Apps and processes
         */

        let mut apps_context = imp.apps_context.borrow_mut();
        apps_context.refresh(process_data);

        imp.applications.refresh_apps_list(&apps_context);
        imp.processes.refresh_processes_list(&apps_context);

        /*
         *  Gpu
         */
        let gpu_pages = imp.gpu_pages.borrow();
        for ((_, page), mut gpu_data) in gpu_pages.values().zip(gpu_data) {
            let page = page.content().and_downcast::<ResGPU>().unwrap();

            if !gpu_data.nvidia {
                // for non-NVIDIA GPUs, we prefer getting the fractions from the processes because they represent the
                // average usage during now and the last refresh, while gpu_busy_percent is a snapshot of the current
                // usage, which might not be what we want

                let processes_gpu_fraction = apps_context.gpu_fraction(gpu_data.gpu_identifier);
                gpu_data.usage_fraction = Some(f64::max(
                    gpu_data.usage_fraction.unwrap_or(0.0),
                    processes_gpu_fraction.into(),
                ));

                let processes_encode_fraction =
                    apps_context.encoder_fraction(gpu_data.gpu_identifier);
                gpu_data.encode_fraction = Some(f64::max(
                    gpu_data.encode_fraction.unwrap_or(0.0),
                    processes_encode_fraction.into(),
                ));

                let processes_decode_fraction =
                    apps_context.decoder_fraction(gpu_data.gpu_identifier);
                gpu_data.decode_fraction = Some(f64::max(
                    gpu_data.decode_fraction.unwrap_or(0.0),
                    processes_decode_fraction.into(),
                ));
            }

            page.refresh_page(&gpu_data);
        }

        /*
         * Npu
         */
        let npu_pages = imp.npu_pages.borrow();
        for ((_, page), mut npu_data) in npu_pages.values().zip(npu_data) {
            let page = page.content().and_downcast::<ResNPU>().unwrap();

            let processes_npu_fraction = apps_context.npu_fraction(npu_data.pci_slot);
            npu_data.usage_fraction = Some(f64::max(
                npu_data.usage_fraction.unwrap_or(0.0),
                processes_npu_fraction.into(),
            ));

            if npu_data.total_memory.is_some() {
                let processes_npu_memory_fraction = apps_context.npu_mem(npu_data.pci_slot);
                npu_data.used_memory = Some(usize::max(
                    npu_data.used_memory.unwrap_or(0),
                    processes_npu_memory_fraction as usize,
                ));
            }

            page.refresh_page(&npu_data);
        }

        std::mem::drop(apps_context);

        /*
         * Cpu
         */
        if let Some(cpu_data) = cpu_data {
            imp.cpu.refresh_page(&cpu_data);
        }

        /*
         * Memory
         */
        if let Some(mem_data_result) = mem_data {
            if let Ok(mem_data) = mem_data_result {
                imp.memory.refresh_page(mem_data);
            } else if let Err(error) = mem_data_result {
                warn!("Unable to update memory data, reason: {error}");
            }
        }

        /*
         *  Drives
         */
        // Make sure there is a page for every drive that is shown
        self.refresh_drive_pages(drive_paths, &drive_data);

        // Update drive pages
        for drive_data in drive_data {
            if drive_data.is_virtual && !SETTINGS.show_virtual_drives() {
                continue;
            }

            let drive_pages = imp.drive_pages.borrow();
            let page = drive_pages.get(&drive_data.inner.sysfs_path).unwrap();
            let page = page.content().and_downcast::<ResDrive>().unwrap();

            page.refresh_page(drive_data);
        }

        /*
         *  Network
         */
        // Make sure there is a page for every network interface that is shown
        self.refresh_network_pages(network_paths, &network_data);

        // Update network pages
        for network_data in network_data {
            if network_data.is_virtual && !SETTINGS.show_virtual_network_interfaces() {
                continue;
            }

            let network_pages = imp.network_pages.borrow();
            let page = network_pages.get(&network_data.inner.sysfs_path).unwrap();
            let page = page.content().and_downcast::<ResNetwork>().unwrap();

            page.refresh_page(network_data);
        }

        /*
         *  Batteries
         */
        // Make sure there is a page for every battery that is shown
        self.refresh_battery_pages(battery_paths, &battery_data);

        // Update battery pages
        for battery_data in battery_data {
            let battery_pages = imp.battery_pages.borrow();
            let page = battery_pages.get(&battery_data.inner.sysfs_path).unwrap();
            let page = page.content().and_downcast::<ResBattery>().unwrap();

            page.refresh_page(battery_data);
        }

        trace!("UI refresh done in {:.2?}", start.elapsed());
    }

    pub async fn periodic_refresh_all(&self) {
        let imp = self.imp();

        let gpus = if ARGS.disable_gpu_monitoring {
            Vec::new()
        } else {
            imp.gpu_pages
                .borrow()
                .values()
                .map(|(gpu, _)| gpu)
                .cloned()
                .collect::<Vec<Gpu>>()
        };

        let npus = if ARGS.disable_npu_monitoring {
            Vec::new()
        } else {
            imp.npu_pages
                .borrow()
                .values()
                .map(|(npu, _)| npu)
                .cloned()
                .collect::<Vec<Npu>>()
        };

        let logical_cpus = imp.cpu.imp().logical_cpus_amount.get();

        let (tx_data, rx_data) = std::sync::mpsc::sync_channel(1);
        let (tx_wait, rx_wait) = std::sync::mpsc::sync_channel(1);

        std::thread::spawn(move || {
            trace!("Spawning refresh thread");

            loop {
                let data = Self::gather_refresh_data(logical_cpus, &gpus, &npus);
                tx_data.send(data).unwrap();

                // Wait on delay so we don't gather data multiple times in a short time span
                // Which usually just yields the same data and makes changes appear delayed by (up to) multiple refreshes
                rx_wait.recv().unwrap();
            }
        });

        let mut first_refresh = true;

        trace!("Going into refresh loop");

        loop {
            // gather_refresh_data()
            let refresh_data = rx_data.recv().unwrap();

            self.refresh_ui(refresh_data);

            // if this is our first refresh, we want to set the opening view to what it was when the last session was
            // ended or whatever the user has supplied via CLI arg
            if first_refresh {
                let page_to_open = ARGS
                    .open_tab_id
                    .clone()
                    .unwrap_or_else(|| SETTINGS.last_viewed_page());

                // yes, this is bad and O(n).
                for page in imp.content_stack.pages().iter::<gtk::StackPage>().flatten() {
                    let toolbar = page.child().downcast::<adw::ToolbarView>().unwrap();

                    let child_id = toolbar.content().unwrap().property::<GString>("tab_id");

                    if child_id == page_to_open {
                        imp.content_stack.set_visible_child(&toolbar);
                        imp.resources_sidebar
                            .set_selected_list_item_by_tab_id(&child_id);
                        break;
                    }
                }

                first_refresh = false;
            }

            // Total time before next ui refresh
            let total_delay = SETTINGS.refresh_speed().ui_refresh_interval();

            // Reasonable timespan before total_delay ends to gather all data
            let gather_time = 0.2;

            timeout_future(Duration::from_secs_f32(total_delay - gather_time)).await;

            // Tell other threads to start gethering data
            tx_wait.send(()).unwrap();

            timeout_future(Duration::from_secs_f32(gather_time)).await;
        }
    }

    /// Wrapper to remove page, and check if removed page was visible with global default behavior
    fn remove_page(&self, page: &ToolbarView) {
        let imp = self.imp();

        // no visible child exists
        if let Some(visible_child) = imp.content_stack.visible_child() {
            if visible_child == *page.upcast_ref::<gtk::Widget>() {
                imp.resources_sidebar
                    .set_selected_list_item_by_tab_id(applications::TAB_ID);
            }
        }

        imp.content_stack.remove(page);
    }

    /// Create page for every drive that is shown
    fn refresh_drive_pages(&self, mut paths: Vec<PathBuf>, drive_data: &[DriveData]) {
        trace!("Refreshing drive pages…");

        let imp = self.imp();

        let mut drive_pages = imp.drive_pages.borrow_mut();

        let mut highest_secondary_ord = drive_pages
            .values()
            .filter_map(adw::ToolbarView::content)
            .map(|widget| widget.property::<u32>("secondary_ord"))
            .max()
            .unwrap_or_default();

        let old_page_paths: Vec<PathBuf> = drive_pages
            .iter()
            .map(|(path, _)| path.to_owned())
            .collect();

        // Filter hidden drives
        for data in drive_data {
            if data.is_virtual && !SETTINGS.show_virtual_drives() {
                let idx = paths
                    .iter()
                    .position(|p| **p == data.inner.sysfs_path)
                    .unwrap();
                paths.remove(idx);
            }
        } // paths now contains all the (paths to) drives we want to show

        // Delete hidden old drive pages
        for page_path in &old_page_paths {
            if !paths.contains(page_path) {
                // A drive has been removed
                info!(
                    "A drive has been removed (or turned invisible): {}",
                    page_path.display()
                );

                let page = drive_pages.remove(page_path).unwrap();
                self.remove_page(&page);
            }
        }

        // Add new drive pages
        for path in paths {
            drive_pages.entry(path.clone()).or_insert_with(|| {
                // A drive has been added
                info!(
                    "A drive has been added (or turned visible): {}",
                    path.display()
                );

                highest_secondary_ord = highest_secondary_ord.saturating_add(1);

                let drive = drive_data
                    .iter()
                    .find(|d| d.inner.sysfs_path == path)
                    .unwrap();

                let display_name = drive.inner.display_name();

                let page = ResDrive::new();
                page.init(drive, highest_secondary_ord);

                if let Some(model) = &drive.inner.model {
                    self.add_page(&page, model, &display_name)
                } else {
                    self.add_page(&page, &drive.inner.block_device, &display_name)
                }
            });
        }
    }

    /// Create page for every network interface that is shown
    fn refresh_network_pages(&self, mut paths: Vec<PathBuf>, network_data: &[NetworkData]) {
        trace!("Refreshing network pages…");

        let imp = self.imp();

        let mut network_pages = imp.network_pages.borrow_mut();

        let mut highest_secondary_ord = network_pages
            .values()
            .filter_map(adw::ToolbarView::content)
            .map(|widget| widget.property::<u32>("secondary_ord"))
            .max()
            .unwrap_or_default();

        let old_page_paths: Vec<PathBuf> = network_pages
            .iter()
            .map(|(path, _)| path.to_owned())
            .collect();

        // Filter hidden networks
        for data in network_data {
            if data.is_virtual && !SETTINGS.show_virtual_network_interfaces() {
                let idx = paths
                    .iter()
                    .position(|p| **p == data.inner.sysfs_path)
                    .unwrap();
                paths.remove(idx);
            }
        } // paths now contains all the (paths to) network interfaces we want to show

        // Delete hidden old network pages
        for page_path in &old_page_paths {
            if !paths.contains(page_path) {
                // A network interface has been removed
                info!(
                    "A network interface has been removed (or turned invisible): {}",
                    page_path.display()
                );

                let page = network_pages.remove(page_path).unwrap();
                self.remove_page(&page);
            }
        }

        // Add new network pages
        for path in paths {
            network_pages.entry(path.clone()).or_insert_with(|| {
                // A network interface has been added
                info!(
                    "A network interface has been added (or turned visible): {}",
                    path.display()
                );

                highest_secondary_ord = highest_secondary_ord.saturating_add(1);

                let network_interface = network_data
                    .iter()
                    .find(|d| d.inner.sysfs_path == path)
                    .unwrap();

                // Insert stub page, values will be updated in refresh_page()
                let page = ResNetwork::new();
                page.init(network_interface, highest_secondary_ord);

                self.add_page(
                    &page,
                    &network_interface.inner.display_name(),
                    &network_interface.inner.interface_type.to_string(),
                )
            });
        }
    }

    /// Create page for every battery that is shown
    fn refresh_battery_pages(&self, paths: Vec<PathBuf>, battery_data: &[BatteryData]) {
        trace!("Refreshing battery pages…");

        let imp = self.imp();

        let mut battery_pages = imp.battery_pages.borrow_mut();

        let mut highest_secondary_ord = battery_pages
            .values()
            .filter_map(adw::ToolbarView::content)
            .map(|widget| widget.property::<u32>("secondary_ord"))
            .max()
            .unwrap_or_default();

        let old_page_paths: Vec<PathBuf> = battery_pages
            .keys()
            .map(std::borrow::ToOwned::to_owned)
            .collect();

        // Delete hidden old battery pages
        for page_path in &old_page_paths {
            if !paths.contains(page_path) {
                // A battery has been removed
                info!("A battery has been removed: {}", page_path.display());

                let page = battery_pages.remove(page_path).unwrap();
                self.remove_page(&page);
            }
        }

        // Add new battery pages
        for path in paths {
            battery_pages.entry(path.clone()).or_insert_with(|| {
                // A battery has been added
                info!("A battery has been added: {}", path.display());

                highest_secondary_ord = highest_secondary_ord.saturating_add(1);

                let battery = battery_data
                    .iter()
                    .find(|d| d.inner.sysfs_path == path)
                    .unwrap();

                // Insert stub page, values will be updated in refresh_page()
                let page = ResBattery::new();
                page.init(battery, highest_secondary_ord);

                self.add_page(
                    &page,
                    &battery
                        .inner
                        .sysfs_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy(),
                    &battery.inner.display_name(),
                )
            });
        }
    }

    fn process_action(&self, action: Action) {
        let apps_context = self.imp().apps_context.borrow();
        match action {
            Action::ManipulateProcesses(action, pids, toast_overlay) => {
                let mut processes_unsuccessful: usize = 0;

                let mut first_process = None;

                for (i, pid) in pids.iter().enumerate() {
                    if let Some(process) = apps_context.get_process(*pid) {
                        if i == 0 {
                            first_process = Some(process);
                        }
                        if process.execute_process_action(action).is_err() {
                            processes_unsuccessful += 1;
                        }
                    }
                }

                let toast_message = if processes_unsuccessful > 0 {
                    if pids.len() == 1 {
                        if let Some(display_name) =
                            first_process.map(|process| &process.display_name)
                        {
                            get_named_action_failure(action, display_name)
                        } else {
                            // this should never happen
                            get_action_failure(action, 1)
                        }
                    } else {
                        get_action_failure(action, processes_unsuccessful)
                    }
                } else if pids.len() == 1 {
                    if let Some(display_name) = first_process.map(|process| &process.display_name) {
                        get_action_success(action, display_name)
                    } else {
                        // this should never happen
                        get_processes_success(action, 1)
                    }
                } else {
                    get_processes_success(action, pids.len())
                };

                toast_overlay.add_toast(Toast::new(&toast_message));
            }

            Action::ManipulateApp(action, id, toast_overlay) => {
                let app = apps_context.get_app(&Some(id.clone())).unwrap();
                let result = app.execute_process_action(&apps_context, action);

                let processes_tried = result.len();
                let processes_successful = result.iter().flatten().count();
                let processes_unsuccessful = processes_tried - processes_successful;

                let toast_message = if processes_unsuccessful > 0 {
                    get_action_failure(action, processes_unsuccessful)
                } else {
                    get_action_success(action, &app.display_name)
                };

                toast_overlay.add_toast(Toast::new(&toast_message));
            }

            Action::AdjustProcess(pid, niceness, affinity, display_name, toast_overlay) => {
                if let Some(process) = apps_context.get_process(pid) {
                    let result = process.adjust(niceness, affinity);

                    let toast_message = match result {
                        Ok(()) => i18n_f("Successfully adjusted {}", &[&display_name]),
                        Err(_) => i18n_f("There was a problem adjusting {}", &[&display_name]),
                    };
                    toast_overlay.add_toast(Toast::new(&toast_message));
                }
            }
        };
    }

    fn save_window_size(&self) -> Result<(), glib::BoolError> {
        let (width, height) = self.default_size();

        SETTINGS.set_window_width(width)?;
        SETTINGS.set_window_height(height)?;

        SETTINGS.set_maximized(self.is_maximized())?;

        Ok(())
    }

    fn load_window_size(&self) {
        let width = SETTINGS.window_width();
        let height = SETTINGS.window_height();
        let is_maximized = SETTINGS.maximized();

        self.set_default_size(width, height);

        if is_maximized {
            self.maximize();
        }
    }

    fn add_page(
        &self,
        widget: &impl IsA<Widget>,
        window_title: &str,
        window_subtitle: &str,
    ) -> adw::ToolbarView {
        trace!("Adding page {window_title} ({window_subtitle})…");

        let imp = self.imp();

        let title_widget = adw::WindowTitle::new(window_title, window_subtitle);

        let sidebar_button = gtk::ToggleButton::new();
        sidebar_button.set_icon_name("sidebar-show-symbolic");
        imp.split_view
            .bind_property("collapsed", &sidebar_button, "visible")
            .sync_create()
            .build();
        imp.split_view
            .bind_property("show-sidebar", &sidebar_button, "active")
            .sync_create()
            .bidirectional()
            .build();

        let header_bar = adw::HeaderBar::new();
        header_bar.add_css_class("flat");
        header_bar.set_title_widget(Some(&title_widget));
        header_bar.pack_start(&sidebar_button);

        let toolbar = adw::ToolbarView::new();
        toolbar.add_top_bar(&header_bar);
        toolbar.set_content(Some(widget));

        imp.content_stack.add_child(&toolbar);

        toolbar
    }
}

impl Default for MainWindow {
    fn default() -> Self {
        Application::default()
            .active_window()
            .unwrap()
            .downcast()
            .unwrap()
    }
}

fn get_action_success(action: ProcessAction, name: &str) -> String {
    match action {
        ProcessAction::TERM => i18n_f("Successfully ended {}", &[name]),
        ProcessAction::STOP => i18n_f("Successfully halted {}", &[name]),
        ProcessAction::KILL => i18n_f("Successfully killed {}", &[name]),
        ProcessAction::CONT => i18n_f("Successfully continued {}", &[name]),
    }
}

fn get_processes_success(action: ProcessAction, count: usize) -> String {
    match action {
        ProcessAction::TERM => ni18n_f(
            "Successfully ended the process",
            "Successfully ended {} processes",
            count as u32,
            &[&count.to_string()],
        ),
        ProcessAction::STOP => ni18n_f(
            "Successfully halted the process",
            "Successfully halted {} processes",
            count as u32,
            &[&count.to_string()],
        ),
        ProcessAction::KILL => ni18n_f(
            "Successfully killed the process",
            "Successfully killed {} processes",
            count as u32,
            &[&count.to_string()],
        ),
        ProcessAction::CONT => ni18n_f(
            "Successfully continued the process",
            "Successfully continued {} processes",
            count as u32,
            &[&count.to_string()],
        ),
    }
}

fn get_action_failure(action: ProcessAction, count: usize) -> String {
    match action {
        ProcessAction::TERM => ni18n_f(
            "There was a problem ending a process",
            "There were problems ending {} processes",
            count as u32,
            &[&count.to_string()],
        ),
        ProcessAction::STOP => ni18n_f(
            "There was a problem halting a process",
            "There were problems halting {} processes",
            count as u32,
            &[&count.to_string()],
        ),
        ProcessAction::KILL => ni18n_f(
            "There was a problem killing a process",
            "There were problems killing {} processes",
            count as u32,
            &[&count.to_string()],
        ),
        ProcessAction::CONT => ni18n_f(
            "There was a problem continuing a process",
            "There were problems continuing {} processes",
            count as u32,
            &[&count.to_string()],
        ),
    }
}

pub fn get_named_action_failure(action: ProcessAction, name: &str) -> String {
    match action {
        ProcessAction::TERM => i18n_f("There was a problem ending {}", &[name]),
        ProcessAction::STOP => i18n_f("There was a problem halting {}", &[name]),
        ProcessAction::KILL => i18n_f("There was a problem killing {}", &[name]),
        ProcessAction::CONT => i18n_f("There was a problem continuing {}", &[name]),
    }
}
