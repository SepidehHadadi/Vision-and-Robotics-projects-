
%*********************************read image*******************************

% [infilename2, path1]=uigetfile('*.jpg','enter the file directory');
%read training database image and load the directory in def


clear all 
close all
clc
% [infilename2, path1]=uigetfile('*.jpg','enter the file directory');

%here there is a utility that help the user to enter the directry of
%training image

prompt='Enter Input Image directory like[F:\computervision\test image]';
def={'images\normalized_images\*.jpg'};
dlgTitle='Input Image File Directory';
lineNo=1;
path=char(inputdlg(prompt,dlgTitle,lineNo,def));
files=dir(path);
s=size(files,1);
%the files are loaded one-by-one
filename=files(1).name;
img1=imread(['images\normalized_images\' filename]);
M=size(img1,1);
N=size(img1,2);
D=zeros(s,M*N);

%***************************read the taining sets construct D matrix************


figure
%lable table
Lt=zeros(1,s);
%read the images and show them then comstruct matrix D
for i=1:s
    filename=files(i).name;
    img=imread(['images\normalized_images\' filename]);
    Lt(1,i)=i;
    %subplot(4,5,i), imshow(img);
    if(size(img,3)>=3)
        img=rgb2gray(img);
    end
    D(i, :) = reshape(img', 1, N*M);
end

%*********************calculate zigma matrix(matrix of covariance)***********

M1=mean(D');
P=D;
for i=1:M*N
    P(:,i)=D(:,i)-M1';
end
Z=(1/(M*N-1))*P*P';
%***************************calculation of phi matrix***********************

[U,S,V]=svd(Z);
sum=0;
for i=1:20
    sum=sum+S(i,i);
end
    
%*****************reading the validating test image************************
    
%in this part the program do the same thing as you did see in the 20th line 
%except here just one file name is requested to find among the data base.

%img2=imread('F:\computer vision\first semester\applied Mathematic\homework\homework two\pca_faces\test_images\3.jpg');
prompt='Enter Input Image directory like[F:\computervision\test image]';
def={'images\normalized_images\*.jpg'};
dlgTitle='Input Image File Directory';
lineNo=1;
imgfile=char(inputdlg(prompt,dlgTitle,lineNo,def));
figure;
m=0;

%***********************central core of face recognition algorithm**********
%in this part of program we start form minimum level of accuracy which set
%approximately around 0.1(10%) and with this assumtion we calculate the number of
%eigenfaces then we chech the extracted image and evluate it. absolutly, 
%in some cases the extracted imagewould be wrong. then we increase the level
%of accuracy till 1.0(100%) and we wil see that 
% finally,be able to find the right answer. in fact, this part of program is
%slightly bit more than what is askeded. it has done to show the
%sensitivity of algorithm with respect ot the accuracy level.

error=zeros(1,10);
for w=1:20
    m=0.05+m;
    sumi=0;
    for i=1:20
        sumi=S(i,i)+sumi;
        if((sumi/sum)>m)
            j=i;
            break;
        end
    end
    
%****************calculation of Phi*****************************************
%because, MATLAB regularly announce the memory errore so using the first
%technicque of calculation was not led to response. in order to find the
%answer I used the second method

    Up=zeros(20,j);
    for i=1:j
        Up(:,i)=U(:,i);
    end
    phi=D'*Up;
    img2=imread(imgfile);
    img3=rgb2gray(img2);
    I1= reshape(img3', 1, N*M);
    I3=double(I1)*phi;
    E=zeros(1,20);
    
 %***********************Database image projection*************************
 % after the calculation of phi we project all the images to the eigenspace
 % and tain the system to be ready for new feature extraction.
 % simultanously we project the input image to the eigenspace and compare
 % it with the other member of data base.
 
    for i=1:20
        filename=files(i).name;
        imgp=imread(filename);
        imgp=rgb2gray(imgp);
        I4= reshape(imgp', 1, N*M);
        I2=double(I4)*phi;
        E(1,i)=norm(I2-I3);
    end
  %*****************************finding the label of new image**************
  %here based on the minimum difference between the feature of input image
  %and data base image we extract the lable of new image and illustrate it.
  
    min1=min(E);
    indx1=find(E==min1);
    accuracy(w)=min1;
    subplot(2,2,1),plot(1:20,E);
    title('error of ori-DATAB');
    hold on
    subplot(2,2,3)
    imshow(img2);
    title('Input Image')
    subplot(2,2,4)
    imshow(files(indx1).name);
    title('Output Image');
    
  %****************************algorithm monitoring switch****************
  %if you put the pause in the program you could easily monitor the
  %operation of algorithm.I mean you can easily see how the algorithm
  %behave
  
    %pause(1);
end
    subplot(2,2,2),plot(0:19,accuracy)
    title('accuracy-Npca');
    figure;
% ********************eigenface demonstration*****************************
%the eigenface is shown in this section
    for i=1:j-3
        subplot(2,5,i), imshow(mat2gray((reshape(phi(:,i),N,M))'));
    end

    
