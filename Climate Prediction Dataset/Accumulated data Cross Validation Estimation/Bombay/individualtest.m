clc
clear all
close all

[traindata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Bombay\accumulated till 2015.xlsx')) ;
[testdata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Bombay\accumulated till 2015.xlsx')) ;
 [N1 D1] = size(traindata);
labels=traindata(:,1);
data=traindata(:,2:D1);

[N D] = size(data);
testlabel=testdata(:,1);
test=testdata(:,2:D1);

model = svmtrain(labels, data, ('-s 0 -t 2 -c 32 -g 0.001953125 -b 1'));

%% K Means Clustering
K = 4;
size1=size(model.SVs);
numObservarations = size1(1);
dimensions = size1(2);

%data = rand([numObservarations dimensions]);
SVs=model.SVs;
sv_indices=model.sv_indices;
sv_coef=model.sv_coef;
datamodel=[sv_coef sv_indices SVs ]
%% cluster
opts = statset('MaxIter', 500, 'Display', 'iter');
[clustIDX, clusters, interClustSum, Dist] = kmeans(datamodel, K, 'options',opts, ...
    'distance','sqEuclidean', 'EmptyAction','singleton', 'replicates',5);

%% plot data+clusters
figure, hold on
scatter3(datamodel(:,1),datamodel(:,2),datamodel(:,3), 50, clustIDX, 'filled')
scatter3(clusters(:,1),clusters(:,2),clusters(:,3), 200, (1:K)', 'filled')
hold off, xlabel('x'), ylabel('y'), zlabel('z')

%% plot clusters quality
figure
[silh,h] = silhouette(datamodel, clustIDX);
avrgScore = mean(silh);


%% Assign data to clusters
% calculate distance (squared) of all instances to each cluster centroid
D = zeros(numObservarations, K);     % init distances
for k=1:K
    %d = sum((x-y).^2).^0.5
    D(:,k) = sum( ((datamodel - repmat(clusters(k,:),numObservarations,1)).^2), 2);
end

% find  for all instances the cluster closet to it
[minDists, clusterIndices] = min(D, [], 2);

% compare it with what you expect it to be
sum(clusterIndices == clustIDX);
%%  class separation
final=[clustIDX datamodel];
sortedA = sortrows(final,1);  %# Sort the rows by the first column
[~,~,uniqueIndex] = unique(sortedA(:,1));

cellA = mat2cell(sortedA,...  
                    accumarray(uniqueIndex(:),1));  %#   into a cell array
         
                
%% Change the support vectors model remove class
% 
% class1=full(cellA{1,2:});
acc=zeros(1,3);
class2=zeros(0,11);
for i=1:K
class2=[cellA{i}];
class2=full(class2);
% SVs=sparse(class2(:,3:size(class2,2)));
model.sv_coef=class2(:,2);
model.sv_indices=class2(:,3);
model.SVs=sparse(class2(:,4:size(class2,2)));
model.totalSV=size(class2,1);
model.nSV=[size(model.sv_coef(find(model.sv_coef > 0))); size(model.sv_coef(find(model.sv_coef < 0)))];

[predict_label, accuracy, prob_values] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
C = confusionmat(testlabel, predict_label) 
accuracy
temp=[accuracy(1) i size(class2,1)];
acc=[acc ; temp];

end
