/// GPU Acceleration Support for ONNX Inference
///
/// This module provides GPU acceleration capabilities for model inference,
/// with automatic fallback to CPU when GPU is unavailable.
use anyhow::Result;

#[cfg(feature = "ai")]
use ort::ExecutionProviderDispatch;

/// GPU execution provider configuration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuProvider {
    /// NVIDIA CUDA provider
    Cuda { device_id: i32 },
    /// DirectML provider (Windows)
    DirectML { device_id: i32 },
    /// CoreML provider (macOS/iOS)
    CoreML { use_cpu_only: bool },
    /// CPU only (fallback)
    Cpu,
}

impl Default for GpuProvider {
    fn default() -> Self {
        Self::Cpu
    }
}

/// GPU configuration and detection
pub struct GpuConfig {
    /// Preferred GPU provider
    pub provider: GpuProvider,
    /// Enable automatic fallback to CPU
    pub enable_fallback: bool,
    /// Log GPU availability and usage
    pub enable_logging: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            provider: GpuProvider::default(),
            enable_fallback: true,
            enable_logging: true,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU config with CUDA
    pub fn with_cuda(device_id: i32) -> Self {
        Self {
            provider: GpuProvider::Cuda { device_id },
            enable_fallback: true,
            enable_logging: true,
        }
    }

    /// Create a new GPU config with DirectML
    pub fn with_directml(device_id: i32) -> Self {
        Self {
            provider: GpuProvider::DirectML { device_id },
            enable_fallback: true,
            enable_logging: true,
        }
    }

    /// Create a new GPU config with CoreML
    pub fn with_coreml(use_cpu_only: bool) -> Self {
        Self {
            provider: GpuProvider::CoreML { use_cpu_only },
            enable_fallback: true,
            enable_logging: true,
        }
    }

    /// Detect available GPU providers
    pub fn detect_available_providers(&self) -> Vec<GpuProvider> {
        let mut providers = Vec::new();

        // Always have CPU as fallback
        providers.push(GpuProvider::Cpu);

        #[cfg(feature = "ai")]
        {
            // Check for CUDA availability
            if Self::is_cuda_available() {
                providers.push(GpuProvider::Cuda { device_id: 0 });
                if self.enable_logging {
                    log::info!("CUDA GPU detected and available");
                }
            }

            // Check for DirectML availability (Windows)
            #[cfg(target_os = "windows")]
            {
                providers.push(GpuProvider::DirectML { device_id: 0 });
                if self.enable_logging {
                    log::info!("DirectML provider available on Windows");
                }
            }

            // Check for CoreML availability (macOS)
            #[cfg(target_os = "macos")]
            {
                providers.push(GpuProvider::CoreML {
                    use_cpu_only: false,
                });
                if self.enable_logging {
                    log::info!("CoreML provider available on macOS");
                }
            }
        }

        if self.enable_logging {
            log::info!("Available GPU providers: {:?}", providers);
        }

        providers
    }

    /// Check if CUDA is available
    #[cfg(feature = "ai")]
    fn is_cuda_available() -> bool {
        // Simple heuristic: check if nvidia-smi exists
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(not(feature = "ai"))]
    fn is_cuda_available() -> bool {
        false
    }

    /// Get execution provider for ONNX Runtime
    #[cfg(feature = "ai")]
    pub fn get_execution_provider(&self) -> Result<Vec<ExecutionProviderDispatch>> {
        let mut providers = Vec::new();

        match &self.provider {
            GpuProvider::Cuda { device_id } => {
                if self.enable_logging {
                    log::info!("Using CUDA GPU (device: {})", device_id);
                }
                providers.push(ExecutionProviderDispatch::CUDA(Default::default()));
            }
            GpuProvider::DirectML { device_id } => {
                if self.enable_logging {
                    log::info!("Using DirectML (device: {})", device_id);
                }
                #[cfg(target_os = "windows")]
                providers.push(ExecutionProviderDispatch::DirectML(Default::default()));
            }
            GpuProvider::CoreML { use_cpu_only } => {
                if self.enable_logging {
                    log::info!("Using CoreML (CPU only: {})", use_cpu_only);
                }
                #[cfg(target_os = "macos")]
                providers.push(ExecutionProviderDispatch::CoreML(Default::default()));
            }
            GpuProvider::Cpu => {
                if self.enable_logging {
                    log::info!("Using CPU execution");
                }
            }
        }

        // Always add CPU as fallback
        if self.enable_fallback && self.provider != GpuProvider::Cpu {
            providers.push(ExecutionProviderDispatch::CPU(Default::default()));
        }

        Ok(providers)
    }

    /// Get execution provider for ONNX Runtime (stub version)
    #[cfg(not(feature = "ai"))]
    pub fn get_execution_provider(&self) -> Result<Vec<String>> {
        Ok(vec!["CPU".to_string()])
    }
}

/// GPU performance statistics
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// Total GPU inferences
    pub total_gpu_inferences: u64,
    /// Total CPU fallback inferences
    pub total_cpu_fallbacks: u64,
    /// Total GPU inference time (ms)
    pub total_gpu_time_ms: u64,
    /// Total CPU inference time (ms)
    pub total_cpu_time_ms: u64,
}

impl GpuStats {
    /// Record a GPU inference
    pub fn record_gpu_inference(&mut self, time_ms: u64) {
        self.total_gpu_inferences += 1;
        self.total_gpu_time_ms += time_ms;
    }

    /// Record a CPU fallback
    pub fn record_cpu_fallback(&mut self, time_ms: u64) {
        self.total_cpu_fallbacks += 1;
        self.total_cpu_time_ms += time_ms;
    }

    /// Get average GPU inference time
    pub fn avg_gpu_time_ms(&self) -> f64 {
        if self.total_gpu_inferences == 0 {
            0.0
        } else {
            self.total_gpu_time_ms as f64 / self.total_gpu_inferences as f64
        }
    }

    /// Get average CPU inference time
    pub fn avg_cpu_time_ms(&self) -> f64 {
        if self.total_cpu_fallbacks == 0 {
            0.0
        } else {
            self.total_cpu_time_ms as f64 / self.total_cpu_fallbacks as f64
        }
    }

    /// Get GPU speedup factor
    pub fn gpu_speedup(&self) -> f64 {
        let avg_cpu = self.avg_cpu_time_ms();
        let avg_gpu = self.avg_gpu_time_ms();

        if avg_gpu == 0.0 {
            1.0
        } else {
            avg_cpu / avg_gpu
        }
    }

    /// Get GPU usage percentage
    pub fn gpu_usage_percentage(&self) -> f64 {
        let total = self.total_gpu_inferences + self.total_cpu_fallbacks;
        if total == 0 {
            0.0
        } else {
            (self.total_gpu_inferences as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.provider, GpuProvider::Cpu);
        assert!(config.enable_fallback);
        assert!(config.enable_logging);
    }

    #[test]
    fn test_gpu_provider_variants() {
        let cuda = GpuProvider::Cuda { device_id: 0 };
        let directml = GpuProvider::DirectML { device_id: 1 };
        let coreml = GpuProvider::CoreML {
            use_cpu_only: false,
        };
        let cpu = GpuProvider::Cpu;

        assert_ne!(cuda, directml);
        assert_ne!(directml, coreml);
        assert_ne!(coreml, cpu);
    }

    #[test]
    fn test_gpu_stats() {
        let mut stats = GpuStats::default();

        stats.record_gpu_inference(10);
        stats.record_gpu_inference(20);
        stats.record_cpu_fallback(50);

        assert_eq!(stats.total_gpu_inferences, 2);
        assert_eq!(stats.total_cpu_fallbacks, 1);
        assert_eq!(stats.avg_gpu_time_ms(), 15.0);
        assert_eq!(stats.avg_cpu_time_ms(), 50.0);
        assert!((stats.gpu_speedup() - 3.333).abs() < 0.01);
        assert!((stats.gpu_usage_percentage() - 66.666).abs() < 0.01);
    }
}
