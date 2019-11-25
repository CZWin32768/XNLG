from mnt.common import io as mntio

in_fn = ""
out_fn = ""

with open(out_fn, "w") as fp:

  for line in mntio.lines(in_fn):
    if line == "": continue
    tok = line.split("\t")[0]
    if any([c in "qwertyuioplkjhgfdsazxcvbnm" for c in tok]):
      continue
    fp.write(line + "\n")

