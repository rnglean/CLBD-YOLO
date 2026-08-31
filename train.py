from ultralytics import YOLO
import argparse


def main(args):
    model = YOLO("models/clbd-yolo.yaml")

    model.train(
        data=args.data,
        epochs=300,
        imgsz=640,
        batch=32,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=100,
        seed=0,
        deterministic=True,
        workers=4,
        amp=True,
        device=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the TeaBud dataset YAML file.",
    )
    args = parser.parse_args()
    main(args)
