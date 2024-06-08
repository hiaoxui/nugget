# Nugget: Efficient text representation

This package provides a unified solution to "nuggify" encoder-decoder, decoder-only,
and encoder-only (experimental) transformers.

The functions of this package are:
- Given the hidden states of the transformer, score tokens and sub-select nuggets
- Build residual connection from the scores to decoder (or encoder, for encoder-only)

Supported models:
- Bart/MBart
- BERT (experimentatal)
- LLaMA
- T5

Supporting new models should not be hard for standard huggingface transformers.
Refer to the `adaptor` folder to adapt to new models.

# Difference between this package and the published papers

## Residual connection

In papers, the residual connection directly adds nugget scores to the attention weights

$$a_{i,j} = (\mathbf{W}^Q \mathbf{x}_i)^\top(\mathbf{W}^K \mathbf{z}_j) + s_j$$

Where $\mathbf{x}_i$ is the target token, $\mathbf{z}_j$ is a nugget, and $s_j$ is the score for the nugget.
Note this directly affects the *forward pass* of the transformer computation.

In this implementation, we by default use *straight-through estimator*, which is

$$a_{i,j} = (\mathbf{W}^Q \mathbf{x}_i)^\top(\mathbf{W}^K \mathbf{z}_j) + s_j - \mathtt{StopGrad}(s_j)$$

which cancels the effect of $s_j$ on the forward pass.

## Less intrusive implementation

The original implementation of Nugget implements the residual connection inherits huggingface/transformers models 
to pass the `scores` argument. Thus it depends on specific huggingface/transformers versions. In this package, the
implementation only modifies the forward method of the `XFormerAttention` classes. 

The passing of the `scores` argument isn't through "forward" method. Instead, a special class `NuggetScoreFeeder` handles 
it through the context. See usage below.

# Usage

Here is an example on using it on Bart:

```python3
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nugget import nuggify

model_name = 'facebook/bart-base'
ratio = 0.1

model, tok = AutoModelForSeq2SeqLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
inputs = tok('hello world', return_tensors='pt')

scorer, encoder, decoder = nuggify(model)

encoder_out = encoder(**inputs)
nuggets = scorer(**inputs, hidden_states=encoder_out.last_hidden_state, nugget_ratio=ratio)

with scorer.score_context(nuggets):
    decoder_out = decoder(
        encoder_outputs=[nuggets.encoding],  labels=inputs['input_ids'], attention_mask=nuggets.mask,
        decoder_attention_mask=inputs['attention_mask'],
    )

decoder_out.loss.backward()
```

# Load from checkpoint

With T5/Bart as an example.

```python3
from nugget import nuggify
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# load from ckpt and construct the Nugget modules
ckpt = torch.load('/path/to/checkpoint')
model_name, nugget_kwargs = ckpt['model_name'], ckpt['nugget_kwargs']
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
scorer, encoder, decoder, nugget_kwargs = nuggify(base_model, **nugget_kwargs)
scorer.load_state_dict(ckpt['weight'])
# Optionally fix the scorer
scorer.requires_grad_(False)

# example usage
tok = AutoTokenizer.from_pretrained(model_name)
text = 'Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval. It is primarily concerned with giving computers the ability to support and manipulate human language. It involves processing natural language datasets, such as text corpora or speech corpora, using either rule-based or probabilistic (i.e. statistical and, most recently, neural network-based) machine learning approaches. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. To this end, natural language processing often borrows ideas from theoretical linguistics. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.'
inputs = tok(text, return_tensors='pt')
# T5/Bart encoder outputs
encodings = encoder(**inputs)
# subselection happens. The output is Nugget type.
# check the comments of [Nuggets](nugget/utils/types.py) for more information about Nugget
nuggets = scorer(**inputs, hidden_states=encodings.last_hidden_state)

# encoding is the subselected encoding. mask is necessary for a batch of sequences.
print(nuggets.encoding.shape)
print(nuggets.mask.shape)

```


# Instruction on stand-alone scorer loading for decoder-only LMs

A nugget scorer scores each token and generate a scalar value to indicate its priority to be selected as nuggets.
It is a transformer encoder (first k-layers) stacked with an FFN layer.

To load a scorer alone, you need the pretrained transformer checkpint and the scorer checkpoint.
Say we have `meta-llama/Llama-2-7b-chat-hf` and `/path/to/scorer.pkl`, then we

```python3
from nugget import nuggify
from transformers import AutoModelForCausalLM, AutoTokenizer

pretrained = 'meta-llama/Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(pretrained)
tok = AutoTokenizer.from_pretrained(pretrained)

# ratio is nuggets/tokens
scorer, _, _, _ = nuggify(model, ratio=0.1)
scorer.load_scorer('/path/to/scorer')

text = 'Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval. It is primarily concerned with giving computers the ability to support and manipulate human language. It involves processing natural language datasets, such as text corpora or speech corpora, using either rule-based or probabilistic (i.e. statistical and, most recently, neural network-based) machine learning approaches. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. To this end, natural language processing often borrows ideas from theoretical linguistics. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. '
inputs = tok(text, return_tensors='pt')
nuggets = scorer(**inputs)

# scorers selects int(ratio*seq_len) tokens as nuggets. Their indices are
print(nuggets.index)
# if you want to use a different/flexible ratio, you can get the raw scores
print(nuggets.all_scores)
```

Please note the current version of Nugget codebase is tied to huggingface/transformers v4.41.x.

# Citation

Please cite this paper if Nugget is helpful to your research:

```bibtex
@InProceedings{pmlr-v202-qin23a,
  title = 	 {Nugget: Neural Agglomerative Embeddings of Text},
  author =       {Qin, Guanghui and Van Durme, Benjamin},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {28337--28350},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/qin23a/qin23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/qin23a.html},
  abstract = 	 {Embedding text sequences is a widespread requirement in modern language understanding. Existing approaches focus largely on constant-size representations. This is problematic, as the amount of information contained in text often varies with the length of the input. We propose a solution called Nugget, which encodes language into a representation based on a dynamically selected subset of input tokens. These nuggets are learned through tasks like autoencoding and machine translation, and intuitively segment language into meaningful units. We demonstrate Nugget outperforms related approaches in tasks involving semantic comparison. Finally, we illustrate these compact units allow for expanding the contextual window of a language model (LM), suggesting new future LMs that can condition on significantly larger amounts of content.}
}
```

Other related papers that uses the idea of Nugget are [Dodo](https://gqin.me/files/24papers/dodo.pdf) and [STAR](https://doi.org/10.48550/arXiv.2402.01172). 
