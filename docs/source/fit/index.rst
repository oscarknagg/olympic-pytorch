fit()
=====

The ``olympic.fit`` function is the heart of this library and where all the good stuff happens. The aim of this function
is to avoid "hand-rolling" your own training loops and hence present a much cleaner interface like Keras or
Scikit-learn.


The pseudocode for ``fit`` is very simple.::

    def fit(model, optimiser, loss_fn, epochs, dataloader, callbacks, update_fn, update_fn_kwargs):

        callbacks.on_train_begin()

        for epoch in range(1, epochs+1):
            callbacks.on_epoch_begin(epoch)

            epoch_logs = dict()
            for batch_index, batch in enumerate(dataloader):
                batch_logs = dict(batch=batch_index)

                callbacks.on_batch_begin(batch_index, batch_logs)

                x, y = prepare_batch(batch)

                loss, y_pred = update_fn(model, optimiser, loss_fn, x, y, epoch, **update_fn_kwargs)
                batch_logs['loss'] = loss.item()

                # Loops through all metrics
                batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

                callbacks.on_batch_end(batch_index, batch_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()

The default ``update_fn`` is just a regular gradient descent step (see below) but any callable with the right signature
can be passed. Alternate ``update_fn``s could be more involved such as adversarial training or the Model-Agnostic
Meta-Learning algorithm. For an example see `fit/usage`.


.. toctree::
   :maxdepth: 2

   usage

.. automodule:: olympic.train
   :members:
   :exclude-members: batch_metrics

