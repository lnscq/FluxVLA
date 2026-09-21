"""Standalone evaluation using RLinf's EmbodiedEvalRunner."""

from .train import prepare_environment, validate_frontend_cfg


def run(cfg):
    from pathlib import Path

    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    from .env_worker import FluxEnvWorker
    from .rlinf_registry import register
    from .rollout_worker import FluxRolloutWorker

    register()
    if not cfg.runner.only_eval:
        raise ValueError('Evaluation entry requires runner.only_eval=true')
    if cfg.runner.resume_dir:
        raise ValueError(
            'Use runner.ckpt_path for evaluation, not runner.resume_dir')
    if cfg.runner.ckpt_path:
        path = Path(cfg.runner.ckpt_path).expanduser().resolve()
        if path.is_dir():
            path = path / 'actor/model_state_dict/full_weights.pt'
        if not path.is_file():
            raise FileNotFoundError(
                f'RLinf full model weights missing: {path}')
        cfg.runner.ckpt_path = str(path)
    validate_frontend_cfg(cfg, evaluation=True)
    cfg = validate_cfg(cfg)
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    groups = {
        name: worker.create_group(cfg).launch(
            cluster,
            name=cfg[name].group_name,
            placement_strategy=placement.get_strategy(name))
        for name, worker in (('rollout', FluxRolloutWorker), ('env',
                                                              FluxEnvWorker))
    }
    runner = EmbodiedEvalRunner(cfg=cfg, **groups)
    runner.init_workers()
    runner.run()
    # RLinf's Ray shutdown hook may mask failure exit codes. Launchers must
    # require this positive completion signal before starting dependent jobs.
    print('FLUXVLA_EVALUATION_COMPLETED', flush=True)


def main():
    import hydra
    import torch.multiprocessing as mp

    from .rlinf_registry import register
    root = prepare_environment()
    register()
    mp.set_start_method('spawn', force=True)

    @hydra.main(
        version_base='1.3',
        config_path=str(root / 'configs/rl'),
        config_name='robotwin_adjust_bottle_eval_fluxvla_pi05')
    def launch(cfg):
        run(cfg)

    launch()


if __name__ == '__main__':
    main()
