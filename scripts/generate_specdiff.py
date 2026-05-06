from generation.specdiff import generate_from_diffusion, parse_args


if __name__ == "__main__":
    args = parse_args()
    generate_from_diffusion(args.checkpoint, args.dataset_path, args.split_path, args.output_path, args.ratio, args.batch_size, args.guidance_scale, args.seed, args.cpu)
