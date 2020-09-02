
README for Convoltion-Neural-Networks Repository
Landon Buell 	2 September 2020
Questions: lhb1007@wildcats.unh.edu

--------------------------------

Required: Python 3.7 or Python 3.8

Python modules/dependencies (usually installing most recent "Anaconda" will work)

	numpy 1.19.1
	tensorflow 2.3.0 (Look into CUDA GPU processing if you have an NVIDIA graphics card)
	pandas 1.1.1
	matplotlib 3.1.0 (Only if you need plotting, otherwise ignore this)

--------------------------------

Python Documentations:

	TensorFlow/Keras
		https://www.tensorflow.org/api_docs
		https://keras.io/
	Numpy:
		https://numpy.org/doc/
	Pandas:
		https://pandas.pydata.org/docs/
		
--------------------------------

Some Resouces that may be helpful:

	Learn about how "Convolutional Layers" work & how to implement them in tensorflow/keras
		https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
		https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848#:~:text=VGG%2D16%20is%20a%20simpler,2%20with%20stride%20of%202.&text=The%20winner%20of%20ILSVRC%202014,also%20known%20as%20Inception%20Module.

	The "bible" of machine learning / neural networks:
		https://www.deeplearningbook.org/

	The "Convolution Layer Group" is visualized here:
		https://neurohive.io/en/popular-networks/vgg16/
		2 adjacent Conv-2D layers, and one "Max/Avg 2D Pooling" layers. Each groups of three is a
			"Convolutional Layer Group"
		"VGG-16" is a very commonly used and effective neural network architecture that may be worth more reading

--------------------------------		