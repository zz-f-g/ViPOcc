import os
from datetime import datetime
from pathlib import Path

import hydra
import ignite
import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast

from utils.array_operations import to
from utils.save_metrics import save_metrics_to_csv


def base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    # output_path = config["output_path"]
    output_path = hydra.core.hydra_config.HydraConfig.get().run.dir
    logger = setup_logger(name=config["name"], filepath=os.path.join(output_path, 'eval.log'))
    log_basic_info(logger, config)

    if rank == 0:
        folder_name = f"{config['name']}"
        if config["stop_iteration"] is not None:
            folder_name += f"_stop-on-{config['stop_iteration']}"

        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output_path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

    # Setup dataflow, model, optimizer, criterion
    test_loader = get_dataflow(config)

    if hasattr(test_loader, "dataset"):
        logger.info(f"Dataset length: Test: {len(test_loader.dataset)}")

    config["num_iters_per_epoch"] = len(test_loader)
    model = initialize(config, logger)

    cp_path = config["checkpoint"]
    if not cp_path.endswith(".pt"):
        cp_path = Path(cp_path)
        cp_path = next(cp_path.glob("training*.pt"))
    checkpoint = torch.load(cp_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = get_metrics(config, device)

    # We define two evaluators as they won't have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(model, metrics=metrics, config=config)

    evaluator.add_event_handler(Events.ITERATION_COMPLETED(every=config["log_every"]), log_metrics_current(logger, metrics))

    try:
        state = evaluator.run(test_loader, max_epochs=1)
        if "tp" in state.metrics and "fp" in state.metrics and "fn" in state.metrics:
            state.metrics["Prec"] = state.metrics["tp"] / (
                state.metrics["tp"] + state.metrics["fp"]
            )
            state.metrics["Rec"] = state.metrics["tp"] / (
                state.metrics["tp"] + state.metrics["fn"]
            )
            state.metrics["IoU"] = state.metrics["tp"] / (
                state.metrics["tp"] + state.metrics["fp"] + state.metrics["fn"]
            )
        log_metrics(logger, state.times["COMPLETED"], "Test", state.metrics)
        logger.info(f"Checkpoint: {str(cp_path)}")
        save_metrics_to_csv(state.metrics, Path(config["metric_save_path"]), config["name"], str(cp_path))
    except Exception as e:
        logger.exception("")
        raise e


def log_basic_info(logger, config):
    logger.info(f"Run {config['name']}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def log_metrics_current(logger, metrics):
    def f(engine):
        out_str = "\n" + "\t".join([f"{v.compute():.3f}".ljust(8) for v in metrics.values()])
        out_str += "\n" + "\t".join([f"{k}".ljust(8) for k in metrics.keys()])
        logger.info(out_str)
    return f


def log_metrics(logger, elapsed, tag, metrics):
    metrics_output = "\n".join([f"{k:15s}: {v:.3f}" for k, v in metrics.items()])
    logger.info(f"\nEvaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n{metrics_output}")


def create_evaluator(model, metrics, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with autocast(enabled=with_amp):
            data = model(data)

        loss_metrics = {}

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {}
        }

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator
