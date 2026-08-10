import argparse
import json

from hstu_kvcache.streaming.kuairand_natural_path_attribution import (
    run_natural_path_attribution,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_natural_path_attribution(parser.parse_args().config)
    if result is not None:
        print(json.dumps(result["parameter_relative_l2_change"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
