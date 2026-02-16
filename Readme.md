# Fine tuning on the Openbookqa
In this repo attempt to fine models on openbookqa was done. In addition, to check forgetting, model were tested with Arc-Easy and Arc-challenge
## Qwen2.5 - 1.5B parameters
### In Lora_ft.ipynb you can find the scripts for fine tuning Qwen. We will fine tune in a classification style, i.e masking final answer only. The steps are done as follows:
* Load model in 4bit
* Load datasets
* Format dataset to format: Question-stem, Choices, Final answer: <Answer Letter>
* Mask formatted dataset with -100 for the final answer (because CrossEntropy ingores -100)
* Collate and pad data
* Initialize and run training with Lora adapters(collect power, time etc.)

### In model_eval.ipynb you can find scripts for evaluating the model on the datasets:
* Load datasets and model
* Format datasets to prompts: Question-stem, Choices, <Answer Letter>
* Ask a model for a Final answer only with prompt
* Generate and parse answer + compare with ground truth, collect time, power etc

## Mamba
Here more general and appropriate approach was chosen to fine tune and evaluate using log-likelihood estimation of the lm-eval-harness framework
* Load datasets and model
* Format training dataset: Question-stem, Answer (unlike with qwen, FT on full Question-Answer pair)
* Collate data
* initialize and run fine-tuning

### Mamba Evaluation
**Based model**

```bash 
lm_eval --model hf     --model_args pretrained=./mamba-1.4b-openbookqa-merged,trust_remote_code=True     --tasks openbookqa,arc_easy,arc_challenge     --device cuda:0     --batch_size 1
```
None, num_fewshot: None, batch_size: 1
Base mamba model evaluation using lm-eval-harness
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.3003|±  |0.0134|
|             |       |none  |     0|acc_norm|↑  |0.3294|±  |0.0137|
|arc_easy     |      1|none  |     0|acc     |↑  |0.6553|±  |0.0098|
|             |       |none  |     0|acc_norm|↑  |0.6124|±  |0.0100|
|openbookqa   |      1|none  |     0|acc     |↑  |0.2600|±  |0.0196|
|             |       |none  |     0|acc_norm|↑  |0.3640|±  |0.0215|

**Fine-Tuned model**
Fine tune script

NOTE: for peft= param use directory where FT model saved
``` bash 
lm_eval --model hf   --model_args pretrained=state-spaces/mamba-1.4b-hf,peft=./mamba-ft,trust_remote_code=True,device_map=auto,dtype=float16   --tasks openbookqa,arc_easy,arc_challenge   --device cuda:0   --batch_size 1   --output_path results_mamba_peft_fp16
```

None, num_fewshot: None, batch_size: 1
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.3532|±  |0.0140|
|             |       |none  |     0|acc_norm|↑  |0.3925|±  |0.0143|
|arc_easy     |      1|none  |     0|acc     |↑  |0.6216|±  |0.0100|
|             |       |none  |     0|acc_norm|↑  |0.5930|±  |0.0101|
|openbookqa   |      1|none  |     0|acc     |↑  |0.2940|±  |0.0204|
|             |       |none  |     0|acc_norm|↑  |0.4240|±  |0.0221|

