
Solution: Error-Compensation-Attacks
Project: Error-Compensation-Attacks_v1
Landon Buell - 1 Sept 2020

	This Folder contains a simulation (v1) of the ECA attack scheme

--------------------------------

Raw Data set: CiFar-10 data: https://www.tensorflow.org/datasets/catalog/cifar10

This project simulates ECA by Attack & compensating as the first two hidden layers in the network
An Approximation layer instance is directly in the network layer stack ("ERR_COMP_v1_Utilites.py" - line 119)
A Compensation layer instance is directly in the network layer stack  ("ERR_COMP_v1_Utilites.py" - line 120)

The result is that the every image is approximated & comepensated ony as it is fed into the network.

--------------------------------

For this model, we test the affect of this tampering over multiple layers over "Convolution Groups"
	We track how doing so affects the (i) loss score, (ii) precision score (iii) recall score

	Based on the parameters for this simulation (described below), the appropriate CNN model is created 
	See "ERR_COMP_Utilities.py" - Line 127. The "FOR" loop in line 133 creates however many layer groups 
		are requird in that particular iteration.

	For this particular simulation - APPROXIMATIONS / COMPENSATION ARE APPLIED DIRECTLY
		INTO THE NEURAL NETWORK MODEL. (See line 119,120) The layers are added directly into the model.
		TO remove the layers, comment out the lines

	In this model, the Approximation also contains a time-based trigger condition. 
		Contained in layer's "Call" method - "ERR_COMP_v1.utilities.py" - line 61
		If the boolean condition is "True" The approximation commences, otherwise the data is passed through the layer un affected


--------------------------------

To run more simulations with this:

	In "ERR_COMP_v1_Utilities.py" there is a section " #### VARIABLES #### " at the top of the script
	Here, you can test the number of convolutional layer groups (line 19) & their pixel grouping size:
		Example:
			{'Single_Layer':   [(2,),(3,),(4,),(5,),(6,)],
                  	'Double_Layer':   [(2,2),(3,3),(4,4),(5,5),(6,6)]}
		Will test 1 & 2 convolutional layer groups for (2 x 2) , (3 x 3) , ... up to (6 x 6) pixel groupings

	The indexs of data that will be approximated / compensated (line 27)
		Example:
			approx_index = np.arange(0,6)
		(or)   	approx_index = [0,1,2,3,4,5]
		Will approximate rows & cols 0 - 5 and rows & cols 26 - 31 (first 6 and last 6 rows & cols)
		The program is setup to handel this particular format

	The outputfile name (line 32)
		Name this as you wish (with extension) to decribe the type of simulation 
		Example:
			"baseline.csv", "TEST1,csv" 
		
	The outputfile path (line 34)
		Change this to fit your personal machine
		These last 2 params are joined together via python for you