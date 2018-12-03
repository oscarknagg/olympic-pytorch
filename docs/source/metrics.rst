Metrics
=======

A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied to
the ``olympic.fit()`` function at training time.

A metric function is similar to a loss function, except that the results from evaluating a metric are not used when
training the model.

You can either pass the name of an existing metric, or pass a PyTorch function.

.. automodule:: olympic.metrics
   :members:

Custom Metrics
--------------

Custom metrics can also be passed to ``olympic.fit``. Custom metrics must take ``(y_true, y_pred)`` as arguments and
return a single float as output. You should be able to pass any PyTorch loss function as a custom metric.