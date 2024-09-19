import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(config: DictConfig) -> None:
    if config.algo.name == "me":
        import main_me as main
    elif config.algo.name == "me_es":
        import main_me_es as main
    elif config.algo.name == "pga_me":
        import main_pga_me as main
    elif config.algo.name == "qd_pg":
        import main_qd_pg as main
    elif config.algo.name == "dcg_me":
        import main_dcg_me as main
    elif config.algo.name == "dcrl_me":
        import main_dcrl_me as main
    elif config.algo.name == "ablation_actor":
        import main_ablation_actor as main
    elif config.algo.name == "ablation_ai":
        import main_ablation_ai as main
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    main()
