clc
clear all
close all

%% generate sample data
K = 2;
load svs.mat
size=size(test);
numObservarations = size(1);
dimensions = size(2);

%data = rand([numObservarations dimensions]);
data=full(test);
%% cluster
opts = statset('MaxIter', 500, 'Display', 'iter');
[clustIDX, clusters, interClustSum, Dist] = kmeans(data, K, 'options',opts, ...
    'distance','sqEuclidean', 'EmptyAction','singleton', 'replicates',3);

%% plot data+clusters
figure, hold on
scatter3(data(:,1),data(:,2),data(:,3), 50, clustIDX, 'filled')
scatter3(clusters(:,1),clusters(:,2),clusters(:,3), 200, (1:K)', 'filled')
hold off, xlabel('x'), ylabel('y'), zlabel('z')

%% plot clusters quality
figure
[silh,h] = silhouette(data, clustIDX);
avrgScore = mean(silh);


%% Assign data to clusters
% calculate distance (squared) of all instances to each cluster centroid
D = zeros(numObservarations, K);     % init distances
for k=1:K
    %d = sum((x-y).^2).^0.5
    D(:,k) = sum( ((data - repmat(clusters(k,:),numObservarations,1)).^2), 2);
end

% find  for all instances the cluster closet to it
[minDists, clusterIndices] = min(D, [], 2);

% compare it with what you expect it to be
sum(clusterIndices == clustIDX);
%%  class separation
final=[clustIDX data];
sortedA = sortrows(final,1);  %# Sort the rows by the first column
[~,~,uniqueIndex] = unique(sortedA(:,1));

cellA = mat2cell(sortedA,...  
                    accumarray(uniqueIndex(:),1));  %#   into a cell array
 cellA{:}  %# Display the contents of the cells
% class1=removerows(data,(clustIDX()==1));

