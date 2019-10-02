%% IMDB CREATER
% Xiaohui Zhang, Jan 24th, 2019
% imdb.images.data is a 4D matrix of size: [height, width, channel, NumOfSlice]
% imdb.images.labels is a 4D matrix of size: [height, width, 2, NumOfSlice] 
% ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights 
% (All voxels with a class label is assigned a weight, details in paper)
% imdb.images.set is [1,NumberOfData] vector with entries 1 or 3 indicating 
% which data is for training and validation respectively.
% --------------------------------------------------------------------

% function imdb = getImdb(dataDir)
clc;
clear all;
% read image files in the folder
% define the path to images
% image width and height
nVx = 256;
nVy = 256;

filelist_images = dir('/home/xiaohui8/Downloads/QuickNAT_tensorflow_V2-master/dataset/images/');
NumOfFile_images = length(filelist_images)-2; % number of nii.gz files
NumOfSlice_images = 256; % z-slice number
channel = 1;
filename_images = cell(NumOfFile_images,1);

% read label files in the folder
% define the path to labels
filelist_labels = dir('/home/xiaohui8/Downloads/QuickNAT_tensorflow_V2-master/dataset/labels/');
NumOfFile_labels = length(filelist_labels)-2; % number of nii.gz files
NumOfSlice_labels = 256; % z-slice number
filename_labels = cell(NumOfFile_labels,1);
%(end)

%% Store the data into imdb
for i = 1:NumOfFile_images
    filename_images{i}= fullfile(filelist_images(i+2).folder,filelist_images(i+2).name); 
end

for i = 1:NumOfFile_labels
    filename_labels{i}= fullfile(filelist_labels(i+2).folder,filelist_labels(i+2).name); 
end

% Preallocate memory 
data = zeros(nVy, nVx, channel, NumOfFile_images*NumOfSlice_images) ; % [height, width, channel, NumberOfData]

% encapsulate this part when you work with test data(start)
label = zeros(nVy, nVx, channel, NumOfFile_labels*NumOfSlice_labels) ; % [height, width, channel, NumberOfData]

% Read image
for k = 1:NumOfFile_images
    train_im = niftiread(filename_images{k}) ;
    label_im = niftiread(filename_labels{k}) ;   
    for n = 1:NumOfSlice_images
        % Store images in imdb structure
        data(:,:,1,(k-1)*NumOfSlice_images+n) = single(squeeze(train_im(:,:,n)));
        label(:,:,1,(k-1)*NumOfSlice_labels+n) = single(squeeze(label_im(:,:,n)));
    end
end

% Store results in the imdb struct
imdb.images.data = data ;
imdb.images.label = label ;
% imdb.images.set = set ;

fprintf('\n***** imdb.mat has been created! *****\n');
save('imdbTraining.mat', 'imdb', '-v7.3'); 
