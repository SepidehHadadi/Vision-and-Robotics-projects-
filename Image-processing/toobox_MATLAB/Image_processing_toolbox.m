function varargout = Image_processing_toolbox(varargin)
% IMAGE_PROCESSING_TOOLBOX MATLAB code for Image_processing_toolbox.fig
%      IMAGE_PROCESSING_TOOLBOX, by itself, creates a new IMAGE_PROCESSING_TOOLBOX or raises the existing
%      singleton*.
%
%      H = IMAGE_PROCESSING_TOOLBOX returns the handle to a new IMAGE_PROCESSING_TOOLBOX or the handle to
%      the existing singleton*.
%
%      IMAGE_PROCESSING_TOOLBOX('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGE_PROCESSING_TOOLBOX.M with the given input arguments.
%
%      IMAGE_PROCESSING_TOOLBOX('Property','Value',...) creates a new IMAGE_PROCESSING_TOOLBOX or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Image_processing_toolbox_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Image_processing_toolbox_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
clc
% Edit the above text to modify the response to help Image_processing_toolbox

% Last Modified by GUIDE v2.5 03-May-2017 09:58:36
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Image_processing_toolbox_OpeningFcn, ...
                   'gui_OutputFcn',  @Image_processing_toolbox_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Image_processing_toolbox is made visible.
function Image_processing_toolbox_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Image_processing_toolbox (see VARARGIN)

% Choose default command line output for Image_processing_toolbox
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Image_processing_toolbox wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Image_processing_toolbox_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im im1 im2;
I=[];
 input_type = get(handles.inputTy,'SelectedObject')
 selected_inputtype=get(input_type,'String')
 %% load and display image
if(selected_inputtype =='Image')
        [path,user_cance]=imgetfile();
        if user_cance
            msgbox(sprintf('Error'),'Error','Error');
             return
        end
        im=imread(path);
        I=im2double(im);%convert image to dobule
        im2=im;%for backup process
        im1=im;
        %axes(hObject);
      axes(handles.axes1);
       imshow(I)
end

%% logo
if(selected_inputtype =='Video')
     load cellsequence
     axes(handles.axes1);
     m=size(cellsequence,3);
     i=1;
     for i=1:m
         imshow(cellsequence(:,:,i))
         pause(0.1)
     end
end
% if(selected_inputtype =='Live')
%     tempImg =zeros(1280,720,3);
%     vid=videoinput('winvideo',1);
% %     handles.vid=videoinput('winvideo',1);
% %    set(handles.vid,'TimerPeriod',0.05,...
% %        'TimerFcn',['if(~isempty(gcf);'...
% %        'image(getsnapshot(handles.vid));'...
% %        'set(handles.axes1,' 'ytick' ',[],' 'xtick' ',[]),'...
% %        'else'...
% %        'delete(imaqfind);'...
% %        'end']);
% %    triggerconfig(handles.vid,'manual')
% %    handles.vid.FramesPerTrigger=Inf;
% %hImage = image(tempImg,'Parent','handles.axes1')
% axes(handles.axes1)
%    preview(vid)
%     %preview(vid,'handles.axes1');
% end
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

input_action = get(handles.actions,'SelectedObject')
 selected_action=get(input_action,'String')
%%
global im1 im2;
if(strcmp(selected_action, 'colorspace'))
    axes(handles.axes2);
    I2=rgb2gray(im2double(im2));
    imshow(I2)
%% add salt and pepper noise
elseif(strcmp(selected_action, 'Salt&pepper'))
    I3=rgb2gray(im2double(im1));
    I30= imnoise(I3,'salt & pepper',0.02);
    axes(handles.axes2);
    imshow(I30)
%% add logo
elseif(strcmp(selected_action, 'Add logo'))
     [path_logo,user_cance_logo]=imgetfile();
        if user_cance_logo
            msgbox(sprintf('Error'),'Error','Error');
             return
        end
    logo=imread(path_logo);
    I4=im2double(im1);
    logo_sized= im2double(imresize(logo,0.4));
    if(size(I4,3)~=0)
        s1=size(logo_sized)
        I4(1:s1(1),1:s1(2),1:3)=logo_sized;
    else
        s1=size(logo_sized)
        I4(1:s1(1),1:s1(2))=reg2gray(logo_sized);
    end
    axes(handles.axes2);
    imshow(I4)
%% compute histogram
elseif(strcmp(selected_action, 'Compute Hist'))
    axes(handles.axes2);
    h1=imhist(rgb2gray(im1));
    bar(h1)
%% histogram equalization
elseif(strcmp(selected_action, 'Hist Equalize'))
    I5 = adapthisteq(rgb2gray(im1),'clipLimit',0.02,'Distribution','rayleigh');
    h2=imhist(I5);
    axes(handles.axes2);
    imshow(I5)
    pause(10)
    bar(h2,'b')
%% Blur image
elseif(strcmp(selected_action, 'Blur'))
    LEN = 21;
    THETA = 11;
    PSF = fspecial('motion', LEN, THETA);
    I6 = imfilter(rgb2gray(im1), PSF, 'conv', 'circular');
    axes(handles.axes2);
    imshow(I6); 
%% edge detection canny
elseif(strcmp(selected_action, 'Canny edge'))
    I7 = edge(rgb2gray(im1),'canny');
    axes(handles.axes2);
    imshow(I7); 
%% laplas operator 
elseif(strcmp(selected_action, 'Laplas'))
    alpha = 0.2;
    H = fspecial('laplacian', alpha)
    I8= imfilter(rgb2gray(im1),H,'replicate');
    axes(handles.axes2);
    imshow(I8); 
%% sobel operator
elseif(strcmp(selected_action, 'Sobel'))
    I9 = edge(rgb2gray(im1),'sobel')
    axes(handles.axes2);
    imshow(I9); 
end
%% advamce option sharpening image
Advance_action = get(handles.AddOps,'SelectedObject')
 selected_Add_action=get(Advance_action,'String')
 if(strcmp(selected_Add_action,'Sharpen'))
    alpha1 =str2double(get(handles.alpha,'String'))
    H=(1/(alpha1+1))*[-alpha1 alpha1-1 -alpha1;alpha1-1 alpha1+5 alpha1-1; -alpha1 alpha1-1 -alpha1]
    % H = fspecial('unsharp');
     I10 = imfilter(rgb2gray(im1),H,'replicate');
     axes(handles.axes2);
     imshow(I10); 
 end
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im3 im2;
h=[1 2 1; 0 0 0; -1 -2 -1];
im3=rgb2gray(im2);
im3=conv2(im3,h,'same');
axes(handles.axes2);
imshow(im3)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clear all
run('ocam_calib')
