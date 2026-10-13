import argparse
import json

from .core import ROOT, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["prepare", "classic", "prompt", "smoke", "full", "evaluate", "infer"]
    )
    parser.add_argument("--kind", choices=["classic", "base", "lora"])
    parser.add_argument("--limit", type=int, help="nombre d'exemples par classe ; défaut : tout le test")
    parser.add_argument("--adapter", default="adapter_smoke")
    parser.add_argument("--text")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        print(prepare())
    elif args.command == "classic":
        from .models import classic_train

        classic_train()
    elif args.command == "prompt":
        from .evaluate import select_prompt

        select_prompt()
    elif args.command in ["smoke", "full"]:
        from .models import lora_train

        print(lora_train(smoke=args.command == "smoke"))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate([args.kind] if args.kind else None, args.limit, args.adapter)
    else:
        from .models import Engine

        engine = Engine(args.kind or "classic", args.adapter)
        result = engine.predict(args.text or "My credit card has a charge I did not authorize.")
        save(ROOT / "reports/last_inference.json", result)
        print(json.dumps(result, ensure_ascii=False))
        engine.close()


if __name__ == "__main__":
    main()
