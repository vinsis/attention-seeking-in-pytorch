### attention-seeking-in-pytorch

This repo contains implementation of various forms of attention:
- [Location based attention](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/location_based_attention.py)
- [Content based dot product](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/content_based_dot_attention.py)
- [Content based concatenation](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/content_based_concat_attention.py)
- [Content based general attention](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/content_based_general_attention.py)
- [Pointer networks](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/pointer_network.py)

and finally

- [No attention](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/no_attention.py)

---

### Task to learn

Each of these sequence to sequence models is trained to learn how to sort a shuffled array of numbers from `1` to `N`. The code to generate this data is [here](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/code/loader.py).

There is a considerable improvement if an attention based model is used versus the no attention model.

---

### Organization of code

All the models and the data loader are defined in `code/`.

- Each model is defined in a separate file. The file containing a model also contains `train` and `test` functions which are self-explanatory.

- Output logs are stored under `training_outputs/`

- Attention weights can be visualized using the code in the notebook [Visualizing attention](https://github.com/vinsis/attention-seeking-in-pytorch/blob/master/Visualizing%20attention.ipynb).
