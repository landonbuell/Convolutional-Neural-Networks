
Solution: Error-Compensation-Attacks
Project: Error-Compensation-Attacks_v0
Landon Buell - 1 Sept 2020

	This Folder contains a simulation (v0) related to the WTA Attack Scheme

--------------------------------

Raw Data set: CiFar-10 data: https://www.tensorflow.org/datasets/catalog/cifar10

This project simulates WTA by approximating data as it is passed into the nerual network model

The result is that any data passed into the network is approximated by a mask-vecotor at layer 2 
	"WTA_Utilities.py"

--------------------------------

For this model, we test the affect of this tampering over multiple layers over "Convolution Groups"
	We track how doing so affects the (i) loss score, (ii) precision score (iii) recall score

	Based on the parameters for this simulation (described below), the appropriate CNN model is created 
	See "ERR_COMP_Utilities.py" - Line 127. The "FOR" loop in line 133 creates however many layer groups 
		are requird in that particular iteration.

	For this particular simulation - APPROXIMATIONS / COMPENSATION IS NOT APPLIED DIRECTLY
		INTO THE NEURAL NETWORK MODEL. In Later simulations approx/compensation IS 
		applied directly in the network

--------------------------------

The Approximation Layer

	The "WhatToApproximateLayer" (line 41) is setup to work on (32 x 32 x 3) input images
		To change this shape, you will have to edit line 52 & change the max & min values in the mask vector(lones 53 & 54)
		
		In order to apply the WTALayer ANYWHERE in the network, 
			you will have to change the expected input shape in the method "InitWeights"
			which is in "WTA_Utilities.py" Line 50-59

		Once the shape of the weights, "W" is change or generalizd, the layer can be used to approximate 
			Activations of any shape, and thus be applied anywhere in the network. 

	Depending on the needs, you may need to re-write this layer entirely to account for various layer dimensions.


--------------------------------

To run more simulations with this:

	In "WTA_Utilities.py" there is a section " #### VARIABLES DECLARATIONS #### " at the top of the script
	Here, you can test the number of convolutional layer groups (line 19) & their pixel grouping size:
		Example:
			KERNELSIZES = np.array([2,3,4,5,6])
		Will test 1 convolutional layer groups for (2 x 2) , (3 x 3) , ... up to (6 x 6) pixel groupings

	maskSize (Line 86) can be changed to determine the Length of the mask vector
		I requires an intgerger argument and can be edited directly in line 86 

	The "FrameName" name (line 22)
		Name this as you wish (with extension) to decribe the type of simulation 
		Example:
			"baseline.csv", "TEST1,csv" 
		
	The outputfile path is set manually in "WTA_MAIN.py" lines 24-26