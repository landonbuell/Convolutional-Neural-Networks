
Solution: Error-Compensation-Attacks
Project: Error-Compensation-Attacks_v0
Landon Buell - 1 Sept 2020

	This Folder contains a simulation (v0) related to the ECA attack scheme

--------------------------------

Raw Data set: CiFar-10 data: https://www.tensorflow.org/datasets/catalog/cifar10

This project simulates ECA by Attack & compensating BEFORE each image is fed into the network.
An Approximation layer instance is created & called - it acts on each image sample ("ERR_COMP_MAIN.py" - line 48)
A Compensation layer instance is created & called - it acts on each image sample ("ERR_COMP_MAIN.py" - line 56)

The result is that the entire dataset has been "tampered" with, and then "corrected" before being fed into the network.

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

To run more simulations with this:

	In "ERR_COMP_Utilities.py" there is a section " #### VARIABLES #### " at the top of the script
	Here, you can test the number of convolutional layer groups (line 22) & their pixel grouping size:
		Example:
			{'Single_Layer':   [(2,),(3,),(4,),(5,),(6,)],
                  	'Double_Layer':   [(2,2),(3,3),(4,4),(5,5),(6,6)]}
		Will test 1 & 2 convolutional layer groups for (2 x 2) , (3 x 3) , ... up to (6 x 6) pixel groupings

	The indexs of data that will be approximated / compensated (line 31)
		Example:
			approx_index = np.concatenate((np.arange(0,6),np.arange(26,32)),axis=-1)
		(or)   	approx_index = [0,1,2,3,4,5]
		Will approximate rows & cols 0 - 5 and rows & cols 26 - 31 
		The program is setup to handel this particular format

	The outputfile name (line 28)
		Name this as you wish (with extension) to decribe the type of simulation 
		Example:
			"baseline.csv", "TEST1,csv" 
		
	The outputfile path (line 29)
		Change this to fit your personal machine
		These last 2 params are joined together via python for you