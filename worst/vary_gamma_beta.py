import hydra
from omegaconf import DictConfig
from adbo.constants import CONFIG_PATH


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="vary_gamma_beta")
def run_vary_gamma_beta(config: DictConfig) -> None:
    print(config)


if __name__ == "__main__":
    run_vary_gamma_beta()
