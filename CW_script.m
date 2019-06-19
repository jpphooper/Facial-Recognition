%% SECTION 1: Load face galleries 

% The data has previously been augmented by blurring and changing
% brightness.
    
faceGallery = imageSet('FaceDatasets','recursive');
faceGalleryCNN = imageDatastore('FaceDatasets', 'LabelSource', 'foldernames','IncludeSubfolders',true); % Needs to be in a datastore for CNN

% Randomly take 40 images for each label
faceGallery = partition(faceGallery, 40, 'randomize'); 
[faceGalleryCNN, NA] = splitEachLabel(faceGalleryCNN, 40, 'randomize'); 

% Split 80/20 for training and testing
[trainingImagesCNN, testImagesCNN] = splitEachLabel(faceGalleryCNN,32,'randomized');
[trainingImages, testImages] = partition(faceGallery, 0.8, 'randomized');

% CNN images need to be of size [227 227 3] to work with AlexNet
augTrainingImagesCNN = augmentedImageDatastore([227 227 3],trainingImagesCNN);
augTestImagesCNN = augmentedImageDatastore([227 227 3],testImagesCNN);




%% SECTION 2: Extract Training Features

% THIS PIECE OF CODE WAS TAKEN FROM TUTORIAL CODE AND ADAPTED

% Initialize Feature vectors
SURFFeaturesTrain = zeros(1728, 500);
HOGFeaturesTrain = zeros(1728,20736);
trainingLabels = cell(1728,1);
featureCount = 1;

% Bag of features used for SURF Features
bag = bagOfFeatures(trainingImages);

% Run through every training image in every label extracting SURF and HOG
% Features.
for i=1:size(trainingImages,2)
    for j = 1:trainingImages(i).Count
        I = read(trainingImages(i),j);
        J = rgb2gray(I);
        HOGFeaturesTrain(featureCount,:) = extractHOGFeatures(I);
        SURFFeaturesTrain(featureCount,:) = encode(bag,J);
        trainingLabels{featureCount} = trainingImages(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = faceGallery(i).Description;
end


%% SECTION 3: Train ECOC and RF Classifiers

SURF_ECOC_Classifier = fitcecoc(SURFFeaturesTrain,trainingLabels);
HOG_ECOC_Classifier = fitcecoc(HOGFeaturesTrain,trainingLabels);
SURF_RF_Classifier = TreeBagger(100,SURFFeaturesTrain,trainingLabels,'Method','Classification');
HOG_RF_Classifier = TreeBagger(100,HOGFeaturesTrain,trainingLabels,'Method','Classification');


%% SECTION 4: Train the CNN using Alexnet

% THIS CODE WAS TAKEN FROM THE MATLAB WEBSITE AND ADAPTED

net = alexnet;
layersTransfer = net.Layers(1:end-3);
layers = [
    layersTransfer
    fullyConnectedLayer(54,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augTestImagesCNN, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augTrainingImagesCNN,layers,options);

%% SECTION 5: Extract Features from Test Images

SURFFeaturesTest = zeros(432, 500);
HOGFeaturesTest = zeros(432,20736);
testLabels = cell(432,1);
featureCount = 1;

for i=1:size(testImages,2)
    for j = 1:testImages(i).Count
        I = read(testImages(i),j);
        J = rgb2gray(I);
        HOGFeaturesTest(featureCount,:) = extractHOGFeatures(I);
        SURFFeaturesTest(featureCount,:) = encode(bag,J);
        testLabels{featureCount} = testImages(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = faceGallery(i).Description;
end


%% SECTION 6: Evaluate Classifers

% Confusion Matrices and Accuracies for each classifier

SURF_ECOC_Predictions = predict(SURF_ECOC_Classifier, SURFFeaturesTest);
SURF_ECOC_Confusion = confusionmat(SURF_ECOC_Predictions,testLabels,'Order',personIndex);
SURF_ECOC_Accuracy = trace(SURF_ECOC_Confusion)/numel(testLabels);

SURF_RF_Predictions = predict(SURF_RF_Classifier, SURFFeaturesTest);
SURF_RF_Confusion = confusionmat(SURF_RF_Predictions,testLabels,'Order',personIndex);
SURF_RF_Accuracy = trace(SURF_RF_Confusion)/numel(testLabels);

HOG_ECOC_Predictions = predict(HOG_ECOC_Classifier, HOGFeaturesTest);
HOG_ECOC_Confusion = confusionmat(HOG_ECOC_Predictions,testLabels,'Order',personIndex);
HOG_ECOC_Accuracy = trace(HOG_ECOC_Confusion)/numel(testLabels);

HOG_RF_Predictions = predict(HOG_RF_Classifier, HOGFeaturesTest);
HOG_RF_Confusion = confusionmat(HOG_RF_Predictions,testLabels,'Order',personIndex);
HOG_RF_Accuracy = trace(HOG_RF_Confusion)/numel(testLabels);

[CNN_Predictions,CNN_Scores] = classify(netTransfer,augTestImagesCNN);
CNN_Confusion = confusionmat(CNN_Predictions, testImagesCNN.Labels);
CNN_Accuracy = trace(CNN_Confusion)/numel(testImagesCNN.Labels);


    