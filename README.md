# SIT770 2.2D – Zero-Shot Twitter Sentiment Error Analysis

This repository contains the experimental materials for my SIT770 2.2D research problem formulation task.

## Project Title

**Identifying Which Informal Twitter Text Types Challenge Zero-Shot Sentiment Classification**

## Purpose

The aim of this experiment is to identify which type of informal Twitter text causes the greatest performance degradation for a recent zero-shot sentiment classifier.

The tested Twitter input categories are:

- Short text
- Slang-based text
- Ambiguous/negation-based text

The experiment focuses on Twitter sentiment data because the revised research question is not about broad domain shift. Instead, it investigates which informal Twitter text type is most challenging for a zero-shot sentiment model.

## Model Used

The experiment uses the following zero-shot model:

```text
cross-encoder/nli-deberta-v3-large
```

This model was used through the Hugging Face `zero-shot-classification` pipeline.

The candidate sentiment labels were:

```text
negative
positive
```

The model was not fine-tuned on the TweetEval dataset. It classified each tweet by selecting the most suitable candidate label in a zero-shot setting.

## Dataset

The dataset used in this experiment is:

```text
tweet_eval / sentiment
```

The original TweetEval sentiment labels are:

```text
0 = negative
1 = neutral
2 = positive
```

For this experiment, neutral samples were removed to create a binary sentiment classification task:

```text
negative = 0
positive = 1
```

This allowed the zero-shot model to be evaluated using the candidate labels `negative` and `positive`.

## Experimental Setup

The experiment was run in Google Colab using a Tesla T4 GPU.

Main setup:

- Dataset: TweetEval sentiment
- Test samples: 2,000 binary Twitter samples
- Model: Zero-shot DeBERTa-v3-large NLI model
- Pipeline: Hugging Face `zero-shot-classification`
- Candidate labels: `negative`, `positive`
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Special-case comparison: balanced subsets of 164 tweets per category
- Random seed: 42

## Special-Case Category Rules

The special-case subsets were created using rule-based filters.

### Short Text

Tweets with fewer than 8 words were selected as short text.

### Slang-Based Text

Tweets were selected as slang-based text if they contained one or more predefined informal/slang terms such as:

```text
meh, idk, smh, bruh, ugh, nah, yikes, lmao, rofl, wtf, omg,
lowkey, highkey, deadass, no cap, cap, sus, trash, fire, mid,
this ain't it, not it, big yikes, love that for you, yeah right,
thanks a lot
```

### Ambiguous/Negation-Based Text

Tweets were selected as ambiguous/negation-based text if they contained at least one of the following patterns:

- Negation terms
- Mixed positive and negative sentiment words
- Uncertainty phrases
- Sarcasm-like expressions

Examples include:

```text
not, never, no, nothing, barely, without,
good + bad, love + hate, amazing + terrible,
maybe, perhaps, I guess, kind of, sort of,
yeah right, just perfect, what a joke
```

This category was designed to capture context-dependent sentiment expressions where the meaning may not be clear from simple positive or negative keywords.

## Why Balanced Subsets Were Used

The initial rule-based filters produced different subset sizes because each category used different linguistic criteria.

For example:

- Short text was selected only by word count.
- Slang text was selected using a predefined slang keyword list.
- Ambiguous/negation text combined multiple patterns, including negation, mixed sentiment, uncertainty, and sarcasm-like expressions.

Because the ambiguous/negation category combined several patterns, it naturally produced more samples than the slang category.

To make the final comparison fairer, the special-case evaluation used balanced subsets:

```text
Short: 164 samples
Slang: 164 samples
Ambiguous/Negation: 164 samples
```

The balanced subsets were selected using fixed random seed `42`.

The balanced subset size of 164 was chosen because it was the size of the smallest available special-case subset. This avoids oversampling or duplicating examples.

## Main Results

### Overall Twitter Performance

| Case | Samples | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Overall Twitter | 2000 | 0.901 | 0.891 | 0.844 | 0.867 |

### Balanced Special-Case Results

| Case | Samples | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Short | 164 | 0.878 | 0.939 | 0.869 | 0.903 |
| Slang | 164 | 0.915 | 0.933 | 0.848 | 0.889 |
| Ambiguous/Negation | 164 | 0.915 | 0.816 | 0.816 | 0.816 |

The ambiguous/negation-based category produced the lowest F1-score in the balanced comparison.

This suggests that the zero-shot DeBERTa model handled short and slang-based tweets relatively well, but showed weaker reliability on context-dependent tweets involving negation, uncertainty, mixed sentiment, or sarcasm-like expressions.

## Confusion Matrix

The experiment also generated a confusion matrix for the overall Twitter test set.

The confusion matrix showed:

```text
False positives: 79
False negatives: 120
```

This means the model more often missed positive tweets than incorrectly labelled negative tweets as positive.

The confusion matrix image is included in this repository as:

```text
zeroshot_deberta_confusion_matrix.png
```

## Error Analysis

The experiment also analysed misclassified tweets by tagging them with linguistic patterns.

| Pattern | Error Count |
|---|---:|
| Short | 20 |
| Slang | 14 |
| Negation | 34 |
| Mixed sentiment | 5 |
| Sarcasm-like | 2 |
| Emoji/symbols | 35 |

This supports the finding that context-dependent and symbolic Twitter expressions remain challenging for the zero-shot model.

## Why the Finding Changed from the Earlier Version

In the earlier version of the task, the experiment used a supervised DistilBERT sentiment classifier and found slang-based text to be the weakest category.

In this revised version, the experiment uses a recent zero-shot DeBERTa-v3-large model and a balanced special-case subset design. Under this revised setting, ambiguous/negation-based text produced the weakest F1-score.

This does not mean the earlier result was random. Instead, it shows that different model types can fail on different informal Twitter patterns.

## Files in This Repository

This repository includes:

```text
zero_shot_deberta_experiment.py
Colab-exported experiment PDF
zeroshot_deberta_confusion_matrix.png
README.md
```

## How to Run

1. Open the Python script or Colab-exported experiment file.
2. Install the required libraries:

```python
!pip install -q datasets transformers accelerate sentencepiece scikit-learn seaborn
```

3. Run the script/cells in order.

The experiment will:

- Load the TweetEval sentiment dataset.
- Convert the dataset into binary sentiment by removing neutral samples.
- Load the zero-shot DeBERTa-v3-large model.
- Evaluate overall Twitter sentiment performance.
- Create rule-based short, slang, and ambiguous/negation subsets.
- Balance the special-case subsets to 164 samples each.
- Evaluate accuracy, precision, recall, and F1-score.
- Generate the confusion matrix.
- Perform error-pattern analysis.

## Notes

This experiment is intended as empirical motivation for a research problem formulation task.

It is not presented as a final benchmark-level evaluation. The goal is to show, using controlled evidence, that recent zero-shot sentiment models can still show uneven robustness across different informal Twitter text types.
