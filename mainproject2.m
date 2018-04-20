clc
clear all;
close all;
warning off all;
t = dir('database');
t1 = struct2cell(t);
t2 = t1(1,3:end);
h = waitbar(0,'Reading images and extracting features..');
q=1;
for i =1:length(t2)
    waitbar(i/length(t2))
    im = imresize(imread(strcat('database\',t2{i})),[512,512]);
    I3 = imcrop(im,[80 95 150 120]);
    subplot(2,2,1);
    imshow(im)
    subplot(2,2,2);
    imshow(I3)
    I4 = im2bw(I3,.45);
    subplot(2,2,3);
    imshow(I4)
    [j,k] = size(I4);
    c(i)=0;
    for j=1:120
        for k =1:150
            if I4(j,k)==0
                c(i)= c(i)+1;
            end
        end
    end
    if c>19000
        disp('Hemorrhage'),disp(int2str(i));
    else
        I5 = rgb2hsver(I3);
        subplot(2,2,4);
        imshow(I5);
        m(i) = mean2(I5);
        s(i) = std2(I5);
    end
end
close(h)
p=[c;m;s];
disp(size(p));
% % 01 Represents people suffering 
% % 10 Represents Normal
t = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 ;
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0];
x = p;

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% For help on training function 'trainscg' type: help trainscg
% For a list of all training functions type: help nntrain
net.trainFcn = 'trainscg';  % Scaled conjugate gradient

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};


% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t  .* tr.valMask{1};
testTargets = t  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, plotconfusion(t,y)
figure, plotroc(t,y)
figure, ploterrhist(e)
save('net.mat','net');


