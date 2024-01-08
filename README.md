# LLM Alignment with NLF

This is the repo for our research on LLM Alignment with Natural Language Feedback. In this repo work on a preliminarly step to assess the viability of our approach and use synthetically defined language tags instead of full-blown natural language feedback. The language tags are obtained by leveraging a Reward Model to quantize reward scores assigned to LLM-generared outputs and further mapping the quantiles into tags. Notice that this poses a limitation in our approach, as we are restricted by the performance of the Reward Model and the expressibility of the language tags. However, we achieve SoTA results in an unlearning toxicity task on the RealToxicityPrompts test set.

In this branch, we tackled a single-task reward setting consisting of unlearning toxicity from LLM-generated text and evaluated on the RealToxicityPrompts test set. In a separate 'sentiment' branch we carried out another experiment to assess the generalization capabilities of our approach, by using the toxicity checkpoint and just changing the language tags at inference time, without any further training. We evaluated on the OpenWebText Corpus (OWT) test set.

On the unlearning toxicity task, during training we employed 5 quantiles which we further mapped into the following language tags:

python´´
tags = [
        "Lowest Toxicity",
        "Low-Moderate Toxicity",
        "Moderate Toxicity",
        "High-Moderate Toxicity",
        "Maximum Toxicity"
    ]
´´
Then, at inference time, we conditioned on the tag associated with the highest-reward quantile, i.e., "Lowest Toxicity".

## Model Checkpoint

We release our model checkpoint for the unlearning toxicity task [here]([gdrive_link](https://drive.google.com/file/d/1x8Y5HMTrcekLdb0hP2pp3Wy3kICN9FDU/view?usp=sharing)).

## Requirement
We suggest using conda to setup environment. You need to first replace ``prefix`` in [environment.yml](environment.yml) with your home path. With conda installed, create an environment called `nlf` with:
```
conda env create -f environment.yml
```

We use the [PerspectiveAPI](https://github.com/conversationai/perspectiveapi) to score toxicity in reward computing, which requires API key for access.
Please refer to their website for API key application. Replace `PERSPECTIVE_API_KEY` in [constants.py](utils/constants.py) with your own API key.

### Training

For training for toxicity reduction with default hyperparameters,
```
python main.py
```
You can change hyperparameters in [arguments.py](arguments.py) via argparse.

### Evaluation

To evaluate the toxicity of unlearned model, please use [sample.py](sample.py). You need to first replace ``save_path`` and ``checkpoint_path`` with your output directory and model checkpoint path (can be provided through CLI), then
```
python sample.py
```
It will save the evaluation result to your output directory.

To evaluate perplexity of the generations, please use [perplexity.py](perplexity.py). You need to first replace ``save_path`` (can be provided through CLI) with the same output directory specified above, then
```
python perplexity.py
```
It will save the perplexity result to the same output directory.




