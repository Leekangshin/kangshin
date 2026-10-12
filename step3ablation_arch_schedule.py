from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

import step3ablationf as base


ORIGINAL_GENERATOR = base.UNetGenerator
ORIGINAL_SCHEDULE = base.set_train_hyperparams


class ReducedSkip4Generator(ORIGINAL_GENERATOR):
    """Keep four shallow skip paths while preserving the baseline parameter count."""

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, torch.zeros_like(d7))
        u2 = self.u2(u1, torch.zeros_like(d6))
        u3 = self.u3(u2, torch.zeros_like(d5))
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)


def constant_schedule(epoch, args, optimizer_g, optimizer_d):
    del epoch
    for group in optimizer_g.param_groups:
        group["lr"] = args.lr_g
    if optimizer_d is not None:
        for group in optimizer_d.param_groups:
            group["lr"] = args.lr_d
    return args.lr_g, args.lr_d, args.lambda_l1


def build_args(variant: str, nf: int, epochs: int, batch_size: int, num_workers: int, resume: bool):
    args = base.parse_args(nf)
    args.epochs = epochs
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.resume = resume
    if variant == "reduced_skip4":
        args.model = "cgan_reducedskip4"
        args.variant = "four_shallow_skips"
        args.active_skip_connections = 4
        args.disabled_skip_connections = ["d7_to_u1", "d6_to_u2", "d5_to_u3"]
        args.training_schedule = "two_stage_baseline"
    elif variant == "one_stage":
        args.model = "cgan_onestage"
        args.variant = "constant_schedule"
        args.active_skip_connections = 7
        args.disabled_skip_connections = []
        args.training_schedule = "constant_lr_and_l1_for_300_epochs"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return args


def configure_variant(variant: str):
    if variant == "reduced_skip4":
        base.UNetGenerator = ReducedSkip4Generator
        base.set_train_hyperparams = ORIGINAL_SCHEDULE
    elif variant == "one_stage":
        base.UNetGenerator = ORIGINAL_GENERATOR
        base.set_train_hyperparams = constant_schedule
    else:
        raise ValueError(f"Unknown variant: {variant}")


def dry_run(nf: int):
    with torch.inference_mode():
        baseline = ORIGINAL_GENERATOR(base_nf=nf).eval()
        reduced = ReducedSkip4Generator(base_nf=nf).eval()
        sample = torch.zeros(1, 3, 512, 512)
        baseline_output = baseline(sample)
        reduced_output = reduced(sample)
    baseline_parameters = base.count_parameters(baseline)
    reduced_parameters = base.count_parameters(reduced)
    print(f"baseline output: {tuple(baseline_output.shape)}")
    print(f"reduced output:  {tuple(reduced_output.shape)}")
    print(f"baseline parameters: {baseline_parameters:,}")
    print(f"reduced parameters:  {reduced_parameters:,}")
    if baseline_output.shape != reduced_output.shape:
        raise RuntimeError("Output shapes do not match")
    if baseline_parameters != reduced_parameters:
        raise RuntimeError("Reduced-skip experiment changed the parameter count")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["reduced_skip4", "one_stage", "all"], default="all")
    parser.add_argument("--nf", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_cli()
    if cli.dry_run:
        dry_run(cli.nf)
        return
    variants = ["reduced_skip4", "one_stage"] if cli.variant == "all" else [cli.variant]
    for variant in variants:
        configure_variant(variant)
        args = build_args(
            variant=variant,
            nf=cli.nf,
            epochs=cli.epochs,
            batch_size=cli.batch_size,
            num_workers=cli.num_workers,
            resume=not cli.no_resume,
        )
        print("=" * 90)
        print(f"Architecture/training sensitivity experiment: {variant}")
        print("=" * 90)
        base.train(args)


if __name__ == "__main__":
    main()
