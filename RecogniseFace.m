function P = RecogniseFace(I, featureType, classifierName)

% featureType can be {'SURF','HOG',''}
% classifierName can be {'RF','ECOC','CNN')

% Detect Faces in image to be classified
detector = vision.CascadeObjectDetector;
bbox = step(detector,I);

% Initialize P
P = zeros(size(bbox,1),3);

% For each combination of classifiers and features it runs through each
% image, resizes, classifies and then puts classificain and central postion
% into P.

% Any differences between versions are commented on.

if strcmp(classifierName,'ECOC') == 1
    if strcmp(featureType,'SURF') == 1
        load('bag.mat','bag');
        compactMdl = loadCompactModel('SURF_ECOC_Classifier.mat');
        for i = 1:size(bbox,1)
            subI = rgb2gray(imresize(imcrop(I, bbox(i,:)),[200 200])); % Must be same size as classified images [200 200]
            feature = encode(bag, subI); % SURF Features
            label = predict(compactMdl, feature);
            P(i,1) = str2double(label);
            P(i,2) = round(bbox(i,1) + bbox(i,3)/2); % x centre
            P(i,3) = round(bbox(i,2) + bbox(i,4)/2); % y centre
        end
    elseif strcmp(featureType,'HOG') == 1
        compactMdl = loadCompactModel('HOG_ECOC_Classifier.mat');
        for i = 1:size(bbox,1)
            subI = imresize(imcrop(I, bbox(i,:)),[200 200]);
            feature = extractHOGFeatures(subI); % HOG Features
            label = predict(compactMdl, feature);
            P(i,1) = str2double(label);
            P(i,2) = round(bbox(i,1) + bbox(i,3)/2);
            P(i,3) = round(bbox(i,2) + bbox(i,4)/2);
        end
    end
elseif strcmp(classifierName,'RF') == 1
    if strcmp(featureType,'SURF') == 1
        load('bag.mat','bag');
        load('SURF_RF_Classifier.mat','SURF_RF_Classifier');
        for i = 1:size(bbox,1)
            subI = rgb2gray(imresize(imcrop(I, bbox(i,:)),[200 200]));
            feature = encode(bag, subI);
            label = predict(SURF_RF_Classifier, feature);
            P(i,1) = str2double(label);
            P(i,2) = round(bbox(i,1) + bbox(i,3)/2);
            P(i,3) = round(bbox(i,2) + bbox(i,4)/2);
        end
    elseif strcmp(featureType,'HOG') == 1
        load('HOG_RF_Classifier.mat','HOG_RF_Classifier');
        for i = 1:size(bbox,1)
            subI = imresize(imcrop(I, bbox(i,:)),[200 200]);
            feature = extractHOGFeatures(subI);
            label = predict(HOG_RF_Classifier, feature);
            P(i,1) = str2double(label);
            P(i,2) = round(bbox(i,1) + bbox(i,3)/2);
            P(i,3) = round(bbox(i,2) + bbox(i,4)/2);
        end
    end
elseif strcmp(classifierName, 'CNN') == 1
    if strcmp(featureType, '') == 1
        load('personIndex.mat', 'personIndex');
        load('CNN.mat','netTransfer');
        for i = 1:size(bbox,1)
            subI = imresize(imcrop(I, bbox(i,:)), [227 227]); % must be [227 227] like Alexnet
            label = classify(netTransfer, subI);
            P(i,1) = str2double(personIndex{label}); % preicts from 1 to 54 so need to change to actual labels
            P(i,2) = round(bbox(i,1) + bbox(i,3)/2);
            P(i,3) = round(bbox(i,2) + bbox(i,4)/2);
        end
    end
else
    disp('ERROR: classifierName or featureType incorrect')
end

