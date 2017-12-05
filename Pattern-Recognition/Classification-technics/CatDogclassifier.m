clear all 
close all
clc
%load data
load('dogData.mat')
load ('catData.mat')
%reshape and display datat
for j=1:9
    figure(1);
    subplot(3,3,j);
    dg=reshape(dog(:,j),64,64);
    imshow(dg)
end
for j=1:9
    figure(2);
    subplot(3,3,j);
    ct=reshape(cat(:,j),64,64);
    imshow(ct)
end
%% feature analysis to get the best features
D=double(dog);
C=double(cat);
X=[D C];
[u,s,v]=svd(X,'econ');
figure(3)
plot(diag(s),'ko','Linewidth',[2])
% feature evaluation
figure(4)
for i=1:4
    subplot(2,2,i);
    ef=reshape(u(:,i),64,64);
    pcolor(ef),axis off, shading interp,colormap(hot);
end
figure(5)
plot3(v(1:80,2),v(1:80,3),v(1:80,4),'ko','Linewidth',[2])
hold on
plot3(v(81:end,2),v(81:end,3),v(81:end,4),'ro','Linewidth',[2])
hold off

%% training classifier and testing
q1=randperm(80);
q2=randperm(80);
xdog=v(1:80,2:4);
xcat =v(81:160,2:4);
xtrain=[xdog(q1(1:50),:); xcat(q2(1:50),:)];
xtest=[xdog(q1(51:80),:); xcat(q2(51:80),:)];

theclass =ones(1,100);
theclass(1:50)=-1;
%Train the SVM Classifier
svmStruct = svmtrain(xtrain,theclass,'Kernel_Function','rbf','showplot',true);
% Predict scores over the grid
% d = 0.02;
% [x1Grid,x2Grid] = meshgrid(min(xtrain(:,1)):d:max(xtrain(:,1)),min(xtrain(:,2)):d:max(xtrain(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];
% [~,scores] = predict(cl,xGrid);
%% test the classifier made of SVM
v = svmclassify(svmStruct,xtest,'showplot',true);