Callbacks
=========

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to
get a view on internal states and statistics of the model during training. You can pass a list of callbacks
(as the keyword argument callbacks) to the ``olympic.fit()`` function. The relevant methods of the callbacks will then
be called at each stage of the training

.. automodule:: olympic.callbacks
   :members:
   :exclude-members: CallbackList, DefaultCallback
