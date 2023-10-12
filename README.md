# CTG-NLF

This is the provisional repo for our research on Controllable Text Generation with Natural Language Feedback.

## Requirement
We suggest using conda to setup environment. You need to first replace ``prefix`` in [environment.yml](environment.yml) with your home path. With conda installed, create an environment called `ctgnlf` with:
```
conda env create -f environment.yml
```

We use the [PerspectiveAPI](https://github.com/conversationai/perspectiveapi) to score toxicity in reward computing, which requires API key for access.
Please refer to their website for API key application. Replace `PERSPECTIVE_API_KEY` in [constants.py](utils/constants.py) with your own API key.

### Training

For training CTG-NLF for toxicity reduction with default hyperparameters,
```
python main.py
```
You can change hyperparameters in [arguments.py](arguments.py) via argparse.

### Evaluation

To evaluate the toxicity of unlearned model, please use [sample.py](sample.py). You need to first replace ``save_path`` and ``checkpoint_path`` with your output directory and model checkpoint path, then
```
python sample.py
```
It will save the evaluation result to your output directory.

To evaluate perplexity of the generations, please use [perplexity.py](perplexity.py). You need to first replace ``save_path`` with the same output directory specified above, then
```
python perplexity.py
```
It will save the perplexity result to the same output directory.




