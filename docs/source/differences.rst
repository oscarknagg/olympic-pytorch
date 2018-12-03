Differences between Olympic and Keras
=====================================

fit function instead of fit method. Evaluation is a callback rather than having its own API.

Olympic has a few key differences from Keras.

fit function not fit method()
-----------------------------

This is mostly personal preference as I find this cleaner than creating a ``trainer`` object, "compiling" it and then
calling ``trainer.fit(model)``the ``torchsample`` library does this in order to more closely resemble Keras in which
you must make a ``model.compile`` call.

Evaluation is just another Callback
-----------------------------------

In Keras the evaluation data is passed directly to the ``fit`` or ``fit_generator`` method of a model. However I find
it more consistent to have evaluation on another dataset to be implemented as a Callback.