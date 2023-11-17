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
