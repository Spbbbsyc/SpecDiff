from training.cwgan_gp import DEFAULTS, parse_args, train


if __name__ == "__main__":
    args = parse_args()
    train({**DEFAULTS, **vars(args)})
