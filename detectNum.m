function detectNum(filename)

% Figures are commented out, but are useful to see the process if
% uncommented so they were left in.

[~,~,ext] = fileparts(filename); % Check if video or image
formats = VideoReader.getFileFormats(); % used to see if video is compatible with OS (was running Ubuntu and .mov was not)

if strcmp(ext,'.JPG') == 1 || strcmp(ext,'.jpg') == 1 
    I = imread(filename);
    grayI = rgb2gray(I);
    [mserRegions, mserConnComp] = detectMSERFeatures(grayI,'RegionAreaRange',[200 8000],'ThresholdDelta',4);

    %{
    figure
    imshow(I)
    hold on
    plot(mserRegions, 'showPixelList', true,'showEllipses',false)
    title('MSER regions')
    hold off
    %}
    
    % Use regionprops to measure MSER properties
    mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
    'Solidity', 'Extent', 'Euler', 'Image');

    % Compute the aspect ratio using bounding box data.
    bbox = vertcat(mserStats.BoundingBox);
    w = bbox(:,3);
    h = bbox(:,4);
    aspectRatio = w./h;

    % Threshold the data to determine which regions to remove. These thresholds
    % may need to be tuned for other images.
    filterIdx = aspectRatio' > 3; 
    filterIdx = filterIdx | [mserStats.Eccentricity] > .995 ;
    filterIdx = filterIdx | [mserStats.Solidity] < .3;
    filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
    filterIdx = filterIdx | [mserStats.EulerNumber] < -4;

    % Remove regions
    mserStats(filterIdx) = [];
    mserRegions(filterIdx) = [];

    %{
    % Show remaining regions
    figure
    imshow(I)
    hold on
    plot(mserRegions, 'showPixelList', true,'showEllipses',false)
    title('After Removing Non-Text Regions Based On Geometric Properties')
    hold off
    %}
 
    % Get a binary image of the a region, and pad it to avoid boundary effects
    % during the stroke width computation.
    regionImage = mserStats(6).Image;
    regionImage = padarray(regionImage, [1 1]);

    % Compute the stroke width image.
    distanceImage = bwdist(~regionImage); 
    skeletonImage = bwmorph(regionImage, 'thin', inf);

    strokeWidthImage = distanceImage;
    strokeWidthImage(~skeletonImage) = 0;

    %{
    % Show the region image alongside the stroke width image. 
    figure
    subplot(1,2,1)
    imagesc(regionImage)
    title('Region Image')

    subplot(1,2,2)
    imagesc(strokeWidthImage)
    title('Stroke Width Image')
    %}
    
    % Compute the stroke width variation metric 
    strokeWidthValues = distanceImage(skeletonImage);   
    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
    
    % Threshold the stroke width variation metric
    strokeWidthThreshold = 0.15;
    strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;
    
    % Process the remaining regions
    for j = 1:numel(mserStats)

        regionImage = mserStats(j).Image;
        regionImage = padarray(regionImage, [1 1], 0);

        distanceImage = bwdist(~regionImage);
        skeletonImage = bwmorph(regionImage, 'thin', inf);

        strokeWidthValues = distanceImage(skeletonImage);

        strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

        strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

    end

    % Remove regions based on the stroke width variation
    mserRegions(strokeWidthFilterIdx) = [];
    mserStats(strokeWidthFilterIdx) = [];

    %{
    % Show remaining regions
    figure
    imshow(I)
    hold on
    plot(mserRegions, 'showPixelList', true,'showEllipses',false)
    title('After Removing Non-Text Regions Based On Stroke Width Variation')
    hold off
    %}

    % Get bounding boxes for all the regions
    bboxes = vertcat(mserStats.BoundingBox);
    
    if isempty(bboxes) == 0

        % Convert from the [x y width height] bounding box format to the [xmin ymin
        % xmax ymax] format for convenience.
        xmin = bboxes(:,1);
        ymin = bboxes(:,2);
        xmax = xmin + bboxes(:,3) - 1;
        ymax = ymin + bboxes(:,4) - 1;

        % Expand the bounding boxes by a small amount.
        expansionAmount = 0.05;
        xmin = (1-expansionAmount) * xmin;
        ymin = (1-expansionAmount) * ymin;
        xmax = (1+expansionAmount) * xmax;
        ymax = (1+expansionAmount) * ymax;

        % Clip the bounding boxes to be within the image bounds
        xmin = max(xmin, 1);
        ymin = max(ymin, 1);
        xmax = min(xmax, size(I,2));
        ymax = min(ymax, size(I,1));

        % Show the expanded bounding boxes
        expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
        IExpandedBBoxes = insertShape(I,'Rectangle',expandedBBoxes,'LineWidth',3);
        
        %{
        figure
        imshow(IExpandedBBoxes)
        title('Expanded Bounding Boxes Text')
        %}

        % Compute the overlap ratio
        overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

        % Set the overlap ratio between a bounding box and itself to zero to
        % simplify the graph representation.
        n = size(overlapRatio,1); 
        overlapRatio(1:n+1:n^2) = 0;

        % Create the graph
        g = graph(overlapRatio);

        % Find the connected text regions within the graph
        componentIndices = conncomp(g);

        % Merge the boxes based on the minimum and maximum dimensions.
        xmin = accumarray(componentIndices', xmin, [], @min);
        ymin = accumarray(componentIndices', ymin, [], @min);
        xmax = accumarray(componentIndices', xmax, [], @max);
        ymax = accumarray(componentIndices', ymax, [], @max);

        % Compose the merged bounding boxes using the [x y width height] format.
        textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

        % Remove bounding boxes that only contain one text region
        numRegionsInGroup = histcounts(componentIndices);
        textBBoxes(numRegionsInGroup == 1, :) = [];

        %disp(textBBoxes)
        P = cell(1,size(textBBoxes,1));
        %figure;

        for i = 1:size(textBBoxes,1) % run through each box that is left
            subI = imcrop(I,textBBoxes(i,:)); 
            %subplot(2,2,i)
            %imshow(rgb2hsv(subI));
            HSVImage = rgb2hsv(subI); % Change to HSV
            intensity = HSVImage(:,:,3);
            map= (intensity>=0.9); 
            percentage=sum(map(:))/numel(map)*100; 
            P{i} = percentage; % get percentage of image that is above 0.9 intensity 
        end
        
        % This essentially picks the image that is most white for the white
        % background behind the number.
        
        % this section could be easily changed to filter on a different
        % condition or get rid of it all together to extract all text from
        % all text boxes.
        
        [~, idx] = max(cell2mat(P));
        textBBoxes = textBBoxes(idx,:); 
    end

    % Show the final text detection result.
    ITextRegion = insertShape(I, 'Rectangle', textBBoxes,'LineWidth',3);
    
    %{
    figure
    imshow(ITextRegion)
    title('Detected Text')
    %}
    
    % Finally extract number from the chosen text box
    
    results = ocr(I, textBBoxes, 'CharacterSet','0123456789','TextLayout','Line');
    if isempty({results.Text}) == 0
        a = results.Text;
        if size(a,2) > 2
            b = a(1:2); % This removes the return character sometimes present in OCR.          
        else
            b = a;
        end
    else
        b = 'Number not Found';
    end
    
    disp(b)

elseif any(strcmp(strcat('.',{formats.Extension}),ext)) == 1
    obj = VideoReader(filename);
    duration = obj.Duration;
    Num = cell(1,10);
    for i = 1:10
        obj.CurrentTime = (abs(1-i)*duration)/9.01; % extracts 10 equally spaced frames from the video
        
        % Then it is an almost exactly identical process to one image apart
        % from it takes a vote at the end.
        I = readFrame(obj);
        grayI = rgb2gray(I);
        [mserRegions, mserConnComp] = detectMSERFeatures(grayI,'RegionAreaRange',[200 8000],'ThresholdDelta',4);

        %{
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('MSER regions')
        hold off
        %}

        
        % Use regionprops to measure MSER properties
        mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
        'Solidity', 'Extent', 'Euler', 'Image');

        % Compute the aspect ratio using bounding box data.
        bbox = vertcat(mserStats.BoundingBox);
        w = bbox(:,3);
        h = bbox(:,4);
        aspectRatio = w./h;

        % Threshold the data to determine which regions to remove. These thresholds
        % may need to be tuned for other images.
        filterIdx = aspectRatio' > 3; 
        filterIdx = filterIdx | [mserStats.Eccentricity] > .995 ;
        filterIdx = filterIdx | [mserStats.Solidity] < .3;
        filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
        filterIdx = filterIdx | [mserStats.EulerNumber] < -4;

        % Remove regions
        mserStats(filterIdx) = [];
        mserRegions(filterIdx) = [];

        %{
        % Show remaining regions
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('After Removing Non-Text Regions Based On Geometric Properties')
        hold off
        %}

        % Get a binary image of the a region, and pad it to avoid boundary effects
        % during the stroke width computation.
        regionImage = mserStats(6).Image;
        regionImage = padarray(regionImage, [1 1]);

        % Compute the stroke width image.
        distanceImage = bwdist(~regionImage); 
        skeletonImage = bwmorph(regionImage, 'thin', inf);

        strokeWidthImage = distanceImage;
        strokeWidthImage(~skeletonImage) = 0;

        %{
        % Show the region image alongside the stroke width image. 
        figure
        subplot(1,2,1)
        imagesc(regionImage)
        title('Region Image')

        subplot(1,2,2)
        imagesc(strokeWidthImage)
        title('Stroke Width Image')
        %}

        % Compute the stroke width variation metric 
        strokeWidthValues = distanceImage(skeletonImage);   
        strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

        % Threshold the stroke width variation metric
        strokeWidthThreshold = 0.15;
        strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;

        % Process the remaining regions
        for j = 1:numel(mserStats)

            regionImage = mserStats(j).Image;
            regionImage = padarray(regionImage, [1 1], 0);

            distanceImage = bwdist(~regionImage);
            skeletonImage = bwmorph(regionImage, 'thin', inf);

            strokeWidthValues = distanceImage(skeletonImage);

            strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

            strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

        end

        % Remove regions based on the stroke width variation
        mserRegions(strokeWidthFilterIdx) = [];
        mserStats(strokeWidthFilterIdx) = [];

        %{
        % Show remaining regions
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('After Removing Non-Text Regions Based On Stroke Width Variation')
        hold off
        %}

        % Get bounding boxes for all the regions
        bboxes = vertcat(mserStats.BoundingBox);
        
        if isempty(bboxes) == 0
            

            % Convert from the [x y width height] bounding box format to the [xmin ymin
            % xmax ymax] format for convenience.
            xmin = bboxes(:,1);
            ymin = bboxes(:,2);
            xmax = xmin + bboxes(:,3) - 1;
            ymax = ymin + bboxes(:,4) - 1;

            % Expand the bounding boxes by a small amount.
            expansionAmount = 0.06;
            xmin = (1-expansionAmount) * xmin;
            ymin = (1-expansionAmount) * ymin;
            xmax = (1+expansionAmount) * xmax;
            ymax = (1+expansionAmount) * ymax;

            % Clip the bounding boxes to be within the image bounds
            xmin = max(xmin, 1);
            ymin = max(ymin, 1);
            xmax = min(xmax, size(I,2));
            ymax = min(ymax, size(I,1));

            % Show the expanded bounding boxes
            expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
            IExpandedBBoxes = insertShape(I,'Rectangle',expandedBBoxes,'LineWidth',3);

            %{
            figure
            imshow(IExpandedBBoxes)
            title('Expanded Bounding Boxes Text')
            %}

            % Compute the overlap ratio
            overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

            % Set the overlap ratio between a bounding box and itself to zero to
            % simplify the graph representation.
            n = size(overlapRatio,1); 
            overlapRatio(1:n+1:n^2) = 0;

            % Create the graph
            g = graph(overlapRatio);

            % Find the connected text regions within the graph
            componentIndices = conncomp(g);

            % Merge the boxes based on the minimum and maximum dimensions.
            xmin = accumarray(componentIndices', xmin, [], @min);
            ymin = accumarray(componentIndices', ymin, [], @min);
            xmax = accumarray(componentIndices', xmax, [], @max);
            ymax = accumarray(componentIndices', ymax, [], @max);

            % Compose the merged bounding boxes using the [x y width height] format.
            textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

            % Remove bounding boxes that only contain one text region
            numRegionsInGroup = histcounts(componentIndices);
            textBBoxes(numRegionsInGroup == 1, :) = [];

            P = cell(1,size(textBBoxes,1));
            %figure;
            for j = 1:size(textBBoxes,1)
                subI = imcrop(I,textBBoxes(j,:));
                %subplot(2,2,j)
                %imshow(rgb2hsv(subI));
                HSVImage = rgb2hsv(subI);
                intensity = HSVImage(:,:,3);
                map= (intensity>=0.9);
                percentage=sum(map(:))/numel(map)*100; 
                P{j} = percentage;
            end

            [~, idx] = max(cell2mat(P));
            textBBoxes = textBBoxes(idx,:);
        end

        % Show the final text detection result.
        ITextRegion = insertShape(I, 'Rectangle', textBBoxes,'LineWidth',3);
        
        %{
        figure
        imshow(ITextRegion)
        title('Detected Text')
        %}
        
        results = ocr(I, textBBoxes, 'CharacterSet','0123456789','TextLayout','Line');
        if isempty({results.Text}) == 0
            a = results.Text;
            if size(a,2) > 2
                b = a(1:2); % again solves problem of return character in OCR.
                Num{i} = b;
            else
                Num{i} = a;
            end
        else
            Num{i} = ''; % Stores 10 numbers, 1 for each frame
        end
    end
    y = unique(Num); % Take unique text/numbers
    ids = str2double(y); 
    tokeep = ~isnan(ids); % Keep only if numbers
    y = y(tokeep);
    n = zeros(length(y), 1);
    for k = 1:length(y)
        n(k) = length(find(strcmp(y{k}, Num)));
    end
    [~, itemp] = max(n); % y(itemp) will give the most voted for number.
    if y(itemp) == ""
        disp('Number not found');
    else
        disp(y(itemp))
    end
else 
    disp('Image File is not of the format: .jpg/.JPG')
    disp('OR')
    disp('Video File is not of the format:')
    disp(strcat('.',{formats.Extension})) 
end

