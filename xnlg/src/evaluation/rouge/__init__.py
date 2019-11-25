import os
import tempfile
import shutil
import time
import string
import rouge

from src.evaluation.rouge.bs_pyrouge import Rouge155


# _tok_dict = {"(": "-lrb-", ")": "-rrb-",
#              "[": "-lsb-", "]": "-rsb-",
#              "{": "-lcb-", "}": "-rcb-",
#              "[UNK]": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}
_tok_dict = {"(": "-lrb-", ")": "-rrb-",
             "[": "-lsb-", "]": "-rsb-",
             "{": "-lcb-", "}": "-rcb-",
             "<unk>": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}

def test_rouge(cand, ref):
    temp_dir = tempfile.mkdtemp()
    candidates = cand
    references = ref
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )

def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True

def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)

def process_eval(gold, eval_fn, trunc_len=0, use_rouge=False, zh=False):
    gold_list = []
    with open(gold, "r", encoding="utf-8") as f_in:
        for l in f_in:
            line = l.strip()
            gold_list.append(line)
    if zh:
        gold_list_zh = []
        zh_dict = {}
        for line in gold_list:
            zh_line = []
            line = "".join(line.split(" "))
            for c in line:
                if c in zh_dict:
                    idx = zh_dict[c]
                else:
                    idx = len(zh_dict)
                    zh_dict[c] = idx
                zh_line.append(str(idx))
            gold_list_zh.append(" ".join(zh_line))

    pred_list = []
    with open(eval_fn, "r", encoding="utf-8") as f_in:
        for l in f_in:
            buf = []
            # sentence = fix_tokenization(l.strip()).replace('1', '#')
            # print(sentence)
            sentence = l
            buf.append(sentence)
            if trunc_len > 0:
                num_left = trunc_len
                trunc_list = []
                for bit in buf:
                    tk_list = bit.split()
                    n = min(len(tk_list), num_left)
                    trunc_list.append(' '.join(tk_list[:n]))
                    num_left -= n
                    if num_left <= 0:
                        break
            else:
                trunc_list = buf
            line = "\n".join(trunc_list)
            pred_list.append(line)
    
    if zh:
        pred_list_zh = []
        for line in pred_list:
            line = "".join(line.split(" "))
            zh_line = []
            for c in line:
                if c in zh_dict:
                    idx = zh_dict[c]
                else:
                    idx = "U"
                zh_line.append(str(idx))
            pred_list_zh.append(" ".join(zh_line))

        print(pred_list[0])
        print(gold_list[0])
        gold_list = gold_list_zh
        pred_list = pred_list_zh
                

    # rouge scores
    assert len(pred_list) == len(gold_list)
    
    print(pred_list[0])
    print(gold_list[0])

    if use_rouge:
        evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l'],
            max_n=2,
            limit_length=True,
            length_limit=100,
            length_limit_type='words',
            alpha=0.5, # Default F1_score
            weight_factor=1.2,
            stemming=False)
        eval_res = evaluator.get_scores(pred_list, gold_list)
        print(eval_res)
    else:
        scores = test_rouge(pred_list, gold_list)
        print(scores)

def main(gold, eval_fn, trunc_len=0, use_rouge=False, zh=False):
    process_eval(gold, eval_fn, trunc_len, use_rouge, zh)
  # print(rouge_results_to_str(res))