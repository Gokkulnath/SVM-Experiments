clc
clear all
close all
% loading dataa 
[traindata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Trichy\accumulated till 2014.xlsx')) ;
[testdata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Coimbatore Climate Prediction\Accumulated data Cross Validation Estimation\Trichy\2015e.xlsx')) ;
labels=traindata(:,1); % separate class label 
data=traindata(:,2:end); % separation of remaining data instances
testlabel=testdata(:,1);  % separate class label
test=testdata(:,2:end); % separation of remaining data instances

tlasped=[];
result=[];
% SVdata=cell(5,5);
K = 4; % cluster size
%SamplingInst=50; % Random Sampling Instance Repetition
for iterations =1:10

model = svmtrain(labels, data, ('-s 0 -t 2 -c 32 -g 0.001953125 -b 1')); % determined best parameters using separae algorithm
[predict_labeli, accuracyi, prob_valuesi] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
Ci = confusionmat(testlabel, predict_labeli) 
accuracyi
   %% K Means Clustering
tic;
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

acc=zeros(1,3); % Matrix with Accuaracy  ClusterID and Size of Cluster
class2=zeros(0,11); %Temp Variable for cluster separation and manupulation
for i=1:K
class2=[cellA{i}];
class2=full(class2);
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
acc(~any(acc,2), : ) = []; % to remove zero columns
end
save('cellA.mat','cellA');
%%  2 Terms Accumulation


terms2=zeros(0,11);
acc2=zeros(1,4);
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
AccMax=max(acc3(:,1));
AccMaxlocations=acc3(ismember(acc3,AccMax),2:end);
AccMax=[AccMax AccMaxlocations(1,:)];

% SVdata=[SVdata; acc3;acc2;acc;];
% SVdata=cell(5,5);

K=K+4;
q=toc;
tlasped=[tlasped; q ];
result=[result; AccMax k];
end

result=[result tlasped];