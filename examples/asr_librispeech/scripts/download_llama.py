import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name=None, version_base=None)
def load_and_print_config(cfg: DictConfig):
    """Loads and prints the Hydra configuration in YAML format."""
    print("🔹 Hydra Configuration:")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    load_and_print_config()
