import argparse
import json

from hstu_kvcache.streaming.kuairand_stationary_coordinate_control import (
    render_stationary_coordinate_control,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    result = render_stationary_coordinate_control(parser.parse_args().result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
