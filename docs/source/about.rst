About Olympic PyTorch
=====================

My first foray into deep learning code was Tensorflow. Myself (and many others) found Tensorflow to be powerful but
unwieldy. Next I moved onto Keras, which is a brilliant library that makes deep learning very accessible as
it strips away most of the boilerplate code.

As I started to want more control and to implement research architectures I turned to PyTorch as its dynamic graph and
clean interface made it not only relatively easy to use but also *fun*. However I missed some of the abstractions and
utilities of Keras.

There are other libraries similar to this one (notably ``ignite`` and ``torchsample``) but they weren't quite what I
wanted so I decided to make what I wanted myself. And by make I mean copy and paste from Keras (MIT license)
because don't fix what ain't broken.

Future development
------------------

I only intend to update this library sufficient to keep it compatible with the latest PyTorch and maintain feature
parity with Keras Callbacks. I will not be adding any more features beyond what already exists.
