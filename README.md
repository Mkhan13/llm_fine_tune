# AI Research Code Generation

### Focal Property: AI Research Code Generation

AI reesarch code generation refers to an LLMs ability to automatically generate AI research code given natural language instructions.

We decided to focus on this focal property because:

1. automated code generation can 10x developer/researcher productivity
2. and as a result, boost the pace of AI research

To get started, install the requirements by running `pip install -r requirements.txt`

### Dataset

We are using Algorithmic Research Group's [ArXiVDLInstruct benchmark](https://huggingface.co/datasets/AlgorithmicResearchGroup/ArXivDLInstruct), a instruction-tuning dataset of ArXiV research code that contains natural language instructions, the corresponding code, as well as the full code of the file the function is from.

The dataset contains 700K+ rows, making it a comprehensive dataset for AI research code generation.

To limit the context window of our application, we filter functions with length greater than 1000, and conduct a 80-20 train-test split.

In addition, we train across dataset sizes of 200, 1000, 5000, and 10K samples to determine the relative performance gain of having more samples in the dataset.

### Model

We use [Unsloth's checkpoint](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit) of Meta's Llama3.1-8B-Instruct model. The model is small enough to fit in most standard GPUs but large enough to be generally performant. We also use a 4-bit quantized version to save memory.

### Evaluation Metric

We evaluate our model using two metrics: [CodeJudge](https://arxiv.org/pdf/2410.02184) and [CodeBERTScore](https://arxiv.org/pdf/2302.05527), which are two recent methods proposed in literature for evaluating functional performance in code.

CodeJudge is a LLM-as-a-Judge (GPT-4o backend) that determines the % of samples where the generated code is functionally equivalent to the reference code, while CodeBERTScore is a fine-tuned version of CodeBERT that calculates a F1 score between the generated and reference code.

We chose these metrics above traditional token-based metrics like CodeBLEU or METEOR because these metrics optimize for code similarity, not necessarily functional correctness.

To run evaluation on a fine-tuned model checkpoint, please run `python3 inference_eval.py --api_key OPENAI_API_KEY --subset_size SUBSET_SIZE` for the regular prompt or `python3 inference_eval.py --api_key OPENAI_API_KEY --subset_size SUBSET_SIZE --augmented_prompt` for the augmented prompt. Make sure SUBSET_SIZE is one of the following: (100,200,500,1000,1500,5000,10,000).

### Fine-Tuning Approach

We fine-tune the model using Low Rank Adaptation and the Unsloth framework, which allows us to train the 8B model much faster and with less GPU resources.

We use the famous Stanford-Alpaca prompt template to further instruction-tune the model to follow natural lagnuage instructions from the dataset

In addition, we train on completions only, meaning that we update weights based on predictions on the output only.

We train for 3 epochs across all sample sizes. The fine-tuned model checkpoints can be found [here](https://huggingface.co/moosejuice13) and [here](https://huggingface.co/Violetjy/llms_proj1_500)

To run fine-tuning yourself, navigate to `finetune.ipynb` or `finetune_optimized.ipynb` for the augmented prompt.

### Results

Shown below is a table of results across different subset sizes:
| Sample Size | CodeJudge | CodeBERTScore |
|-------------|-----------|--------------|
| Baseline | 0.085 | 0.554 |
| 200 | 0.15 | 0.674 |
| 500 | 0.16 | 0.681 |
| 1000 | 0.18 | 0.687 |
| 5000 | 0.2 | 0.702 |
| 10,000 | 0.21 | 0.725 |

The reported CodeJudge scores represent proportion of functionally correct code, while CodeBERTScore represents a F1 score between generated and reference code.
