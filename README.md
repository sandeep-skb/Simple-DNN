# Simple-DNN
Creating a 3 layer deep neural net using just numpy
Credit for idea behind the code goes to Siraj Raval https://github.com/llSourcell. My intention behind rewriting the code is to two fold :
1. Hand code using numpy to get better understanding.
2. I wanted to follow the convention of "y = *w.x* + b". I have seen dot product of w and x is not well maintained which leads to confusion of x and w dimensions. Using w.x order allowed me to use the dimension order like (current layer neurons, previous layer neurons)

I have put the dimensions in comments beside each layer in order to demonstrate how the dimensions of W and X are getting used in the W.X format.
