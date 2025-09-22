from code_bert_score import score

def evaluate_codebert_score(examples):
  P, R, F1, F3 = score(examples['predicted'], examples['function'], lang='python', rescale_with_baseline=True, sources=examples['prompt'])
  return {'bertscore': F1}