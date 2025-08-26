import argparse
import pathlib


def main():
    print("[stub] evaluator starting", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["chartqa", "docvqa"], required=True)
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    out_dir = pathlib.Path("vlm/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{args.task}_preds.jsonl"
    preds_path.write_text("")
    print(f"[stub] wrote -> {preds_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
