clc
clear all
close all
% loading dataa 
[traindata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Bombay\2014.xlsx')) ;
[testdata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Bombay\2015.xlsx')) ;
 [N1 D1] = size(traindata);
labels=traindata(:,1); % separate class label 
data=traindata(:,2:D1); % separation of remaining data instances

[N D] = size(data);
testlabel=testdata(:,1);  % separate class label
test=testdata(:,2:D1); % separation of remaining data instances

model = svmtrain(labels, data, ('-s 0 -t 2 -c 32 -g 0.001953125 -b 1')); % determined best parameters using separae algorithm
[predict_labeli, accuracyi, prob_valuesi] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
Ci = confusionmat(testlabel, predict_labeli) 
accuracyi

%% K Means Clustering
K = 10; % cluster size
Instance=1000; %Instance to run
size1=size(model.SVs);  % number of Svs
numObservarations = size1(1);
dimensions = size1(2); % SV Dimension
SVs=model.SVs;  % Temp Variable to store all SVs Found
sv_indices=model.sv_indices;  % Temp Variable to Store Respective Sv indices 
sv_coef=model.sv_coef;   % Temp Variable to Store Respective Sv Coefficients
datamodel=[sv_coef sv_indices SVs ] %data which is to be clustered
%% cluster
opts = statset('MaxIter', 500, 'Display', 'iter');
[clustIDX, clusters, interClustSum, Dist] = kmeans(datamodel, K, 'options',opts, ...
    'distance','sqEuclidean', 'EmptyAction','singleton', 'replicates',5,'start','uniform');

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
acc=zeros(1,3); % Matrix with Accuaracy  ClusterID and Size of Cluster
class2=zeros(0,11); %Temp Variable for cluster separation and manupulation
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
temp=[accuracy(1) i size(class2,1)]; % Accuracy ClusterID Size of Cluster
acc=[acc ; temp]; % Accumulate the accuracy in a single matrix
acc( ~any(acc,2), : ) = []; % to remove zero columns
end
%% Fuzzy Application to get reduced Svs
SvSelect2 = readfis('SvSelect2.fis');
FinalSV=[]; % Matrix to store final support vectors
f=[];% Fuzzy weight output
SvSize =[];Accuracy_Accumulated=[];
for nsampling=1:Instance
    FinalSV=[]; % Matrix to store final support vectors
    f=[];% Fuzzy weight output
for t=1:K
    fout=evalfis(acc(t,1),SvSelect2);
    f=[f;fout];
    SvSize(t)=floor(fout*acc(t,3)); % Whole Number Round Off
    [SampledSVs,SampledSVsidx] = datasample(cellA{t},SvSize(t),1,'Replace',false); % Random sampling based on SvSize
    FinalSV=[FinalSV;SampledSVs];
end

%% Trainning using reduced SVs
FinalSV=full(FinalSV); % Similar to above code trainning with data in FinalSV
model.sv_coef=FinalSV(:,2);
model.sv_indices=FinalSV(:,3);
model.SVs=sparse(FinalSV(:,4:size(FinalSV,2)));
model.totalSV=size(FinalSV,1);
model.nSV=[size(model.sv_coef(find(model.sv_coef > 0))); size(model.sv_coef(find(model.sv_coef < 0)))];
[predict_labelf, accuracyf, prob_valuesf] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
Cf = confusionmat(testlabel, predict_labelf) 
accuracyf
Accuracy_Accumulated=[Accuracy_Accumulated; accuracyf(1)];
end
max(Accuracy_Accumulated)