from generation.cwgan_gp import generate_from_gan, parse_args


if __name__ == "__main__":
    args = parse_args()
    generate_from_gan(args.checkpoint, args.dataset_path, args.split_path, args.output_path, args.ratio, args.seed, args.cpu)
