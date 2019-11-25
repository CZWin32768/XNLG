import argparse

from src.evaluation.rouge import main

parser = argparse.ArgumentParser(description='ROUGE')

parser.add_argument("--ref", type=str, default="",
                      help="ref", required=False)
parser.add_argument("--hyp", type=str, default="",
                    help="hyp", required=True)
parser.add_argument("--zh", default=False,
                    help="hyp")                    
parser.add_argument("--use_rouge", default=False,
                    help="use_rouge")
parser.add_argument("--trunc_len", type=int, default=0,
                    help="trunc_len")
params = parser.parse_args()

main(params.ref, params.hyp, params.trunc_len, params.use_rouge, params.zh)