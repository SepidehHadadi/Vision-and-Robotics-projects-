clear all;
close all;
clc;
%-----------------------------------
load('dogData.mat')
load('catData.mat')
for j=1:9
    figure(1)
    subplot(3,3,j);
    ct=reshape(cat(:,j),64,64);
    imshow(ct)
end


