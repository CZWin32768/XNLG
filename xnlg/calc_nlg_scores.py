import os
import argparse
from nlgeval import compute_metrics


def get_params():
  parser = argparse.ArgumentParser(description='Evaluation')
  parser.add_argument("-i", type=str, required=True)
  parser.add_argument("--dataset_dir", type=str, default="")
  parser.add_argument("--lang", type=str, required=True)
  return parser.parse_args()


def main():
  p = get_params()
  hypothesis=p.i
  references=[os.path.join(p.dataset_dir, "test.q.%s.lc" % p.lang)]
  print(compute_metrics(hypothesis, references, no_overlap=False, no_skipthoughts=True, no_glove=True))


if __name__ == "__main__":
  main()