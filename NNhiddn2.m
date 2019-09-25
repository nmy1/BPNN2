%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 2;  
hidden_layer1_size = 100;
hidden_layer2_size = 100;
labels = 2;

%% Loading
fprintf('Loading Data ...\n')

data=load('data.txt');
X=data(:,1:2);
y=data(:,3);
y=y+1;
m = size(X, 1);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Initializing Pameters
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, labels);




% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

%% Training NN
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 500);
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size...
    + 1))):((hidden_layer1_size * (input_layer_size + 1))...
        + hidden_layer2_size * (hidden_layer1_size + 1))), ...
            hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size...
    + 1))+(hidden_layer2_size * (hidden_layer1_size + 1))):end), ...
        labels, (hidden_layer2_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Implement Predict
pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
