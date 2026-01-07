use anyhow::Result;
use log::info;
use tch::{kind::Kind, nn, nn::ModuleT, Device, Tensor};

/// Simple ML session using tch-rs (PyTorch C++ bindings)
pub struct MlSession {
    device: Device,
    vs: nn::VarStore,
}

impl MlSession {
    /// Create a new session on CPU or CUDA (auto-detect)
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        info!("ML session initialized on {:?}", device);
        Ok(Self { device, vs })
    }

    /// Build a tiny two-layer MLP for demonstration (input_dim -> hidden -> output_dim)
    pub fn tiny_mlp(&self, input_dim: i64, hidden_dim: i64, output_dim: i64) -> impl nn::ModuleT {
        nn::seq_t()
            .add(nn::linear(
                &self.vs.root() / "layer1",
                input_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn_t(|xs, _train| xs.relu())
            .add(nn::linear(
                &self.vs.root() / "layer2",
                hidden_dim,
                output_dim,
                Default::default(),
            ))
    }

    /// Run a single forward pass on random input to validate runtime
    pub fn smoke_test(&self) -> Result<Tensor> {
        let xs = Tensor::rand([4, 8], (Kind::Float, self.device));
        let net = self.tiny_mlp(8, 16, 4);
        let ys = net.forward_t(&xs, /*train=*/ false);
        Ok(ys)
    }
}
