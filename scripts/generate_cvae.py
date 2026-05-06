from generation.cvae import generate_from_cvae, parse_args


if __name__ == "__main__":
    args = parse_args()
    generate_from_cvae(args.checkpoint, args.dataset_path, args.split_path, args.output_path, args.ratio, args.seed, args.cpu)
