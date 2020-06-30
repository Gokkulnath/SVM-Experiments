% /////////////////////////////////////////////////////////////////////////
% //  Code Written and Developed by : Gokkul Nath T S                 /////
% //  Tested with Matlab 2014a (8.3.0.532)  64 Bit                    /////
% //  External Libraries Used : LibSVM 3.20                           /////
% //  System Specification : Intel Core i5 8gb Ram                    /////
% /////////////////////////////////////////////////////////////////////////

clc
clear all
close all
%% Input Data and Folds
[traindata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Bombay\accumulated till 2015.xlsx')) ;
[testdata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Bombay\2015e.xlsx')) ;
labels=traindata(:,1); data=traindata(:,2:end);
testlabel=testdata(:,1); test=testdata(:,2:end);
folds = 10;
K=4; % Cluster Size
[C,gamma] = meshgrid(13:2:17, -13:2:-9);
cv_acc = zeros(numel(C),1);
%% Figure 1 : Conventional SVM :Cross Validation Module and Grid Search

for i=1:numel(C)
    cv_acc(i) = svmtrain(labels, data, ...
                    sprintf('-s 0 -t 2 -c %f -g %f -v %d ', 2^C(i), 2^gamma(i), folds));
end
%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%# now you can train you model using best_C and best_gamma
best_C = 2^C(idx);
best_gamma = 2^gamma(idx);
model = svmtrain(labels, data, ...
                     sprintf('-s 0 -t 2 -c %f -g %f -b 1', best_C, best_gamma));
[predict_labeli, accuracyi, prob_valuesi] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
Ci = confusionmat(testlabel, predict_labeli) 
accuracyi
size(model.SVs)

%% Figure 2 ,3 , 4, 5 : Clustering of Support Vectors: K Means Clustering
tic; % To Measure Clustering Time
% Extraction Of Support Vectors From Conventional Model
size1=size(model.SVs);  numObservarations = size1(1); dimensions = size1(2); 
% Temp Variables 
SVs=model.SVs;  sv_indices=model.sv_indices; sv_coef=model.sv_coef;   
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
sortedA = sortrows(final,1);  %# Sort the rows by the Cluster Index
[~,~,uniqueIndex] = unique(sortedA(:,1));

cellA = mat2cell(sortedA,...  
                    accumarray(uniqueIndex(:),1));  %#   into a cell array

%% Change/Replacement of support vectors in model remove class

acc=zeros(1,3); % Matrix with Accuaracy  ClusterID and Size of Cluster
class2=zeros(0,11); %Temp Variable for cluster separation and manupulation
for i=1:K
class2=[cellA{i}];
class2=full(class2);
model.sv_coef=class2(:,2); model.sv_indices=class2(:,3);  
model.SVs=sparse(class2(:,4:size(class2,2))); model.totalSV=size(class2,1);

model.nSV=[size(model.sv_coef(find(model.sv_coef > 0))); size(model.sv_coef(find(model.sv_coef < 0)))];

[predict_label, accuracy, prob_values] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
C = confusionmat(testlabel, predict_label) 
accuracy
temp=[accuracy(1) i size(class2,1)]; % Accuracy ClusterID Size of Cluster
acc=[acc ; temp]; % Accumulate the accuracy in a single matrix
acc(~any(acc,2), : ) = []; % to remove zero columns
end

%%  2 Terms Accumulation
terms2=zeros(0,11); acc2=zeros(1,4); 
for i=1:K
    for j=2:K
        if(i~=j && j~=i)  
terms2=[cellA(i); cellA(j);];
terms2=cell2mat(terms2);
terms2=full(terms2);
model.sv_coef=terms2(:,2);
model.sv_indices=terms2(:,3);
model.SVs=sparse(terms2(:,4:size(terms2,2)));
model.totalSV=size(terms2,1);
model.nSV=[size(model.sv_coef(find(model.sv_coef > 0))); size(model.sv_coef(find(model.sv_coef < 0)))];
%% 
[predict_label, accuracy, prob_values] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
C = confusionmat(testlabel, predict_label) 
accuracy
i 
j
temp=[accuracy(1) i j  size(terms2,1)];
acc2=[acc2 ; temp];
acc2(~any(acc2,2), : ) = []; % to remove zero columns

end
end
end

%%  Terms 3 Accumulation 
terms3=zeros(0,11);
acc3=zeros(1,5);
for i=1:K
    for j=2:K
       for k=3:K
       if((i~=j) && (j~=k) && (i~=k))  
terms3=[cellA(i); cellA(j); cellA(k);];
terms3=cell2mat(terms3);
terms3=full(terms3);
% SVs=sparse(class2(:,3:size~=(class2,2)));
model.sv_coef=terms3(:,2);
model.sv_indices=terms3(:,3);
model.SVs=sparse(terms3(:,4:size(terms3,2)));
model.totalSV=size(terms3,1);
model.nSV=[size(model.sv_coef(find(model.sv_coef > 0))); size(model.sv_coef(find(model.sv_coef < 0)))];
%% 
[predict_label, accuracy, prob_values] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
C = confusionmat(testlabel, predict_label) 
accuracy
i 
j
k
temp=[accuracy(1) i j k size(terms3,1)];
acc3=[acc3 ; temp];
acc3(~any(acc3,2), : ) = []; % to remove zero columns

       end
       end
    end
end