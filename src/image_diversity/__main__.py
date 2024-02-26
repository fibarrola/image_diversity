from argparse import ArgumentParser
from image_diversity import InceptionMetrics
from image_diversity import ClipMetrics


parser = ArgumentParser()
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Leave blank for auto')
parser.add_argument('--div_type', type=str, default="TCE",
                    choices=["TCE", "TIE"],
                    help=('Diversity measure'))
parser.add_argument('path', type=str,
                    help=('Path to the directory containing the images to be analyzed'))
parser.add_argument('--n_eigs', type=int, default=20,
                    help=('Number of eigenvalues used for computing truncated Entropy'))

def main():
    args = parser.parse_args()

    if args.div_type == "TCE":
        clip_metrics = ClipMetrics(device=args.device, n_eigs=args.n_eigs)
        diversity_score = clip_metrics.tce(args.path)

    elif args.div_type == "TIE":
        inception_metrics = InceptionMetrics(device=args.device, n_eigs=args.n_eigs)
        diversity_score = inception_metrics.tie(args.path)

    print("{div_type} computation finished for {path}... ".format(
        div_type=args.div_type, path=args.path
    ))
    print("{div_type}({n_eigs:0d}) = {div_score:.3f} ".format(
        div_type=args.div_type, n_eigs=args.n_eigs, div_score=diversity_score
    ))


if __name__ == '__main__':
    main()