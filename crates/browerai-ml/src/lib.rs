use anyhow::Result;
use log::info;

#[cfg(feature = "ml")]
use tch::{kind::Kind, nn, nn::ModuleT, Device, Tensor};

#[cfg(feature = "ml")]
pub struct MlSession {
    device: Device,
    vs: nn::VarStore,
}

#[cfg(feature = "ml")]
impl MlSession {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        info!("ML session initialized on {:?}", device);
        Ok(Self { device, vs })
    }

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

    pub fn smoke_test(&self) -> Result<Tensor> {
        let xs = Tensor::rand([4, 8], (Kind::Float, self.device));
        let net = self.tiny_mlp(8, 16, 4);
        let ys = net.forward_t(&xs, false);
        Ok(ys)
    }
}

#[cfg(not(feature = "ml"))]
pub struct MlSession;

#[cfg(not(feature = "ml"))]
impl MlSession {
    pub fn new() -> Result<Self> {
        info!("ML session not available (ml feature not enabled)");
        Ok(Self)
    }

    pub fn tiny_mlp(&self, _input_dim: i64, _hidden_dim: i64, _output_dim: i64) -> MockModule {
        MockModule
    }

    pub fn smoke_test(&self) -> Result<MockTensor> {
        Ok(MockTensor)
    }
}

#[cfg(not(feature = "ml"))]
pub struct MockModule;

#[cfg(not(feature = "ml"))]
impl MockModule {
    pub fn forward_t(&self, _input: &MockTensor, _train: bool) -> MockTensor {
        MockTensor
    }
}

#[cfg(not(feature = "ml"))]
#[derive(Clone)]
pub struct MockTensor;

pub trait MatrixOps {
    fn relu(&self) -> Self;
}

#[cfg(feature = "ml")]
impl MatrixOps for Tensor {
    fn relu(&self) -> Self {
        self.relu()
    }
}

#[cfg(not(feature = "ml"))]
impl MatrixOps for MockTensor {
    fn relu(&self) -> Self {
        self.clone()
    }
}
