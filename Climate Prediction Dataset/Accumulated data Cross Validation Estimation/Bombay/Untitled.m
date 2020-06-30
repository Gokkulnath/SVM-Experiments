opts = statset('Display','final');
[idx,ctrs] = kmeans(test,10,'Distance','city',...
              'Replicates',5,'Options',opts);

plot(test(idx==1,1),test(idx==1,2),'r.','MarkerSize',12)
hold on
plot(test(idx==2,1),test(idx==2,2),'b.','MarkerSize',12)
plot(ctrs(:,1),ctrs(:,2),'kx',...
     'MarkerSize',12,'LineWidth',2)
plot(ctrs(:,1),ctrs(:,2),'ko',...
     'MarkerSize',12,'LineWidth',2)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
hold off