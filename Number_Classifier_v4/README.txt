README.txt for Number_Classifier_v4

Modeling Attack Functions for Forward Propagation in Mutlilayer Perceptron Models
Landon Buell - 7 April 2020

Pythion version 3.8.1
sklearn version 0.22.2

To Run "Control" Models for Number_Classifier_v4, make no adjustments to sklearn.neural_network.MLPClassifier source code. 
This file "_multilayer_perceptron.py". Running and fitting as normal produces and unperturbed control model.

To Simulate an attack with the Number_Classifier_v4 program, 
the follwing lines must be implimented into the "_multilayer_perceptron.py" file course code:

(Lines 33 - 38)

	# IMPORT ATTACK MODULES
	import sys
	attack_path ='C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v4'
	sys.path.insert(1,attack_path)
	from Attack_functions import ATTACK
	# END BUELL IMPORTS 

This imports the ATTACK function from the the Number_Classifier_v4 folder.
Note that 'attack_path' may differ from machine to machine. The current path is setup for Landon's two computers

Additionally, We need to change the forward propoagation matrix mutliplication operation.
This is done in the "def _forward_pass(self, activations):" function (now line 99)