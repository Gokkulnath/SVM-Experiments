 clc
 clear all
close all

[traindata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Coimbatore Climate Prediction\Bombay\traindata.xlsx')) ;
[testdata] = xlsread(fullfile('C:\Users\gokkul nath\Desktop\Coimbatore Climate Prediction\Bombay\testdata2015.xlsx')) ;
 [N1 D1] = size(traindata);


labels=traindata(:,1);
data=traindata(:,2:D1);
[N D] = size(data);

 testlabel=testdata(:,1);
 test=testdata(:,2:D1);


%Scaling of Data 
vec = data(:,:);

%# get max and min
maxVec = max(vec);
minVec = min(vec);

%# normalize to -1...1
for i=1:N
vecN(i,:)=((vec(i,:)-minVec)./(maxVec-minVec) - 0.5 ) *2;


%# to "de-normalize", apply the calculations in reverse
%denormalizedata(i,:) = (vecN(i,:)./2+0.5).* (maxVec-minVec) + minVec
end
folds = 10;
[C,gamma] = meshgrid(-5:2:15, -15:2:3);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
    cv_acc(i) = svmtrain(labels, vecN, ...
                    sprintf('-s 0 -t 2 -c %f -h 0 -g %f -v %d ', 2^C(i), 2^gamma(i), folds));
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%# contour plot of paramter selection
contour(C, gamma, reshape(cv_acc,size(C))), colorbar
hold on
plot(C(idx), gamma(idx), 'rx')
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

%# now you can train you model using best_C and best_gamma
best_C = 2^C(idx);
best_gamma = 2^gamma(idx);


model = svmtrain(labels, data, ...
                    sprintf('-s 0 -t 2 -c %f -g %f -b 1', best_C, best_gamma));
          
[predict_label, accuracy, prob_values] = svmpredict(testlabel, test, model, '-b 1 '); % run the SVM model on the test data
C = confusionmat(testlabel, predict_label) 
accuracy
best_C
