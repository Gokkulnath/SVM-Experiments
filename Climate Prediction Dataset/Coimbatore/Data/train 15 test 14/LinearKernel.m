clc
clear all
close all

[traindata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Climate Prediction\Data\train 15 test 14\traindata.xlsx')) ;
[testdata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\Climate Prediction\Data\train 15 test 14\testdata.xlsx')) ;
% read the data set
% [labels, data] = libsvmread('C:\Users\gokkul nath\Desktop\Ramanathan sir SVM\binary classificaion\a1a\a1a');
[N1 D1] = size(traindata);

labels=traindata(:,1);
data=traindata(:,2:D1);
[N D] = size(data);

testlabel=testdata(:,1);
test=testdata(:,2:D1);



%#  parameters
folds = 5;
C=-15:2:20;
%# Finding best C
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
    cv_acc(i) = svmtrain(labels, data, ...
                    sprintf('-s 0 -t 0 -c %f -v %d ', 2^C(i), folds));
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);
best_C = 2^C(idx);


model = svmtrain(labels, data, ...
                    sprintf('-s 0 -t 0 -c %f -h 0 -b 1', best_C));
                
             
[predict_label, accuracy, prob_values] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data

C = confusionmat(testlabel, predict_label)                   %# confusion matrix

best_C 