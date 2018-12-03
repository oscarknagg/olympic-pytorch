Olympic PyTorch Documentation
=============================

Olympic implements a Keras-like API for PyTorch.

The goal of Olympic is to combine the joy of Pytorch's dynamic graph with the joy of Keras's high level abstractions
for training. Concretely, Olympic contains:

1. The ``olympic.fit()`` function. This implements a very similar API to Keras's ``model.fit`` and
   ``model.fit_generator`` methods in a more functional and less object-oriented fashion and spares you the effort
   from "hand-rolling" your own training loop.
2. ``Callback`` objects that perform functionality common to most deep learning training pipelines such as
   learning rate scheduling, model checkpointing and csv logging. These integrate into ``olympic.fit()`` and spare
   you the effort of writing boilerplate code.
3. Some helpful utility functions such as common metrics and some convenience layers from Keras that are
   missing in PyTorch.


.. toctree::
    :maxdepth: 1
    :caption: Home:

    about
    quickstart
    differences

.. toctree::
    :maxdepth: 2
    :caption: Features:

    fit/index
    callbacks
    layers
    metrics
