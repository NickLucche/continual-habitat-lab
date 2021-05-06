from avalanche_lab.config import AvalancheConfig, base_config
import pytest

def test_default_config_instance():
    base = AvalancheConfig(from_cli=False)
    return base == base_config
