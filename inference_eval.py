from unsloth import FastLanguageModel
from datasets import load_dataset
from prompts import LLAMA_INSTRUCTION_PROMPT, LLAMA_INSTRUCTION_PROMPT_AUGMENTED
from CodeJudge import NaiveJudge
from CodeBERTScore import evaluate_codebert_score
import numpy as np
import argparse


def main(args):
    dataset = load_dataset('AlgorithmicResearchGroup/ArXivDLInstruct', split="train")
    filtered_dataset = dataset.filter(lambda example: len(example["function"]) <= 1000)
    if args.augmented_prompt:
        filtered_dataset = filtered_dataset.filter(lambda example: example['file_length'] <= 5000) ### remove long tail in file lengths

    train_test_split = filtered_dataset.train_test_split(test_size=0.2, seed=42)
    _, test_data = train_test_split["train"], train_test_split["test"]

    test_data = test_data.shuffle(seed=42).select(range(200))

    model_name = f'moosejuice13/llama3_finetune_{args.subset_size}' 
    model_name += '_diff' if args.augmented_prompt else ''
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    def generate_with_hf(example):
        if args.augmented_prompt:
            prompts = LLAMA_INSTRUCTION_PROMPT_AUGMENTED.format(task_description=example['prompt'], input=example['full_code'].replace(example['function'], ''))
        else:
            prompts = LLAMA_INSTRUCTION_PROMPT.format(task_description=example['prompt'])
        inputs = tokenizer(
            [prompts],
            padding_side='left',
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True,
                                temperature = 0.1, min_p = 0.1)
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return {'predicted': generated_texts}

    test_data = test_data.map(generate_with_hf, batched=False)
    judge = NaiveJudge('gpt-4o', args.api_key)
    
    def evaluate_functional_correctness(examples):
        correctness = [judge.evaluate(prompt, predicted, function) for prompt, predicted, function in zip(examples['prompt'], examples['predicted'], examples['function'])]
        return {'correctness': correctness}
    test_data = test_data.map(evaluate_functional_correctness, batched=True, batch_size=16)
    test_data = test_data.map(evaluate_codebert_score, batched=True, batch_size=16)

    print('CodeJudge Score: ', np.mean(test_data['correctness']))
    print('CodeBERTScore: ', np.mean(test_data['bertscore']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--subset_size', type=int, default=1000)
    parser.add_argument('--augmented_prompt', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)