import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import torch
import numpy as np
from omegaconf import OmegaConf


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = OmegaConf.create({
        'model': {
            'target': 'ldm.models.diffusion.ddpm.LatentDiffusion',
            'params': {
                'linear_start': 0.00085,
                'linear_end': 0.0120,
                'num_timesteps_cond': 1,
                'log_every_t': 200,
                'timesteps': 1000,
                'first_stage_key': "image",
                'cond_stage_key': "txt",
                'image_size': 64,
                'channels': 4,
                'cond_stage_trainable': False,
                'conditioning_key': 'crossattn',
                'monitor': 'val/loss_simple_ema',
                'scale_factor': 0.18215,
                'use_ema': False
            }
        },
        'data': {
            'target': 'main.DataModuleFromConfig',
            'params': {
                'batch_size': 4,
                'num_workers': 4,
                'wrap': False
            }
        }
    })
    return config


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_latent():
    """Create a sample latent tensor for testing."""
    return torch.randn(1, 4, 32, 32)


@pytest.fixture
def sample_noise():
    """Create a sample noise tensor for testing."""
    return torch.randn(1, 4, 32, 32)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = torch.device('cpu')
    model.num_timesteps = 1000
    model.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)
    model.log_one_minus_alphas_cumprod = torch.log(1 - model.alphas_cumprod)
    model.sqrt_alphas_cumprod = torch.sqrt(model.alphas_cumprod)
    model.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - model.alphas_cumprod)
    return model


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch = {
        'image': torch.randn(4, 3, 256, 256),
        'txt': ['a cat', 'a dog', 'a bird', 'a fish'],
        'class_label': torch.tensor([0, 1, 2, 3])
    }
    return batch


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([
        {'image': torch.randn(4, 3, 256, 256), 'txt': ['test'] * 4}
        for _ in range(10)
    ]))
    dataloader.__len__ = Mock(return_value=10)
    return dataloader


@pytest.fixture
def numpy_random_seed():
    """Set a fixed random seed for numpy."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def torch_random_seed():
    """Set a fixed random seed for torch."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
    torch.manual_seed(torch.initial_seed())


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint dictionary."""
    checkpoint = {
        'state_dict': {
            'model.diffusion_model.input_blocks.0.0.weight': torch.randn(320, 4, 3, 3),
            'model.diffusion_model.input_blocks.0.0.bias': torch.randn(320),
        },
        'global_step': 100000,
        'epoch': 10,
        'optimizer_states': [{}],
        'lr_schedulers': [{}]
    }
    return checkpoint


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    
    def get_output():
        return captured_output.getvalue()
    
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return get_output