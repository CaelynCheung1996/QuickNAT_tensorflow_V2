%% IMDB CREATER
% Xiaohui Zhang, Jan 24th, 2019
% imdb.images.data is a 4D matrix of size: [height, width, channel, NumberOfData]
% imdb.images.labels is a 4D matrix of size: [height, width, 2, NumberOfData] 
% ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights 
% (All voxels with a class label is assigned a weight, details in paper)
% imdb.images.set is [1,NumberOfData] vector with entries 1 or 3 indicating 
% which data is for training and validation respectively.
% --------------------------------------------------------------------

% function imdb = getImdb(dataDir)
clc;
clear all;
% read image files in the folder
filelist_images = dir('/home/caelyn/Desktop/dataset/OASIS_QuickNAT/training-images/skull-strip/*10*.nii.gz');
%filelist_images = dir('/home/caelyn/Desktop/dataset/OASIS_QuickNAT/testing-images/skull-strip/*1*.nii.gz');
NumOfFile_images = length(filelist_images); % nii.gz number
NumOfData_images = 256; % z-slice number
channel = 1;
filename_images = cell(NumOfFile_images,1);

% read label files in the folder
filelist_labels = dir('/home/caelyn/Desktop/dataset/OASIS_QuickNAT/training-labels-remap-256/*10*.nii.gz');
NumOfFile_labels = length(filelist_labels); % nii.gz number
NumOfData_labels = 256; % z-slice number
channel = 1;
filename_labels = cell(NumOfFile_labels,1);

%% Store the data into imdb
for i = 1:NumOfFile_images
    filename_images{i}= fullfile(filelist_images(i).folder,filelist_images(i).name); 
end

for i = 1:NumOfFile_labels
    filename_labels{i}= fullfile(filelist_labels(i).folder,filelist_labels(i).name); 
end
% Preallocate memory 
data = zeros(256, 256, channel, NumOfFile_images*NumOfData_images) ; % [height, width, channel, NumberOfData]
label = zeros(256, 256, channel, NumOfFile_labels*NumOfData_labels) ; % [height, width, channel, NumberOfData]

% Read image
for k = 1:NumOfFile_images
    train_im = niftiread(filename_images{k}) ;
    label_im = niftiread(filename_labels{k}) ;   
    for n = 1:NumOfData_images
        % Store images in imdb structure
        data(:,:,1,(k-1)*256+n) = single(squeeze(train_im(:,n,:)));
        label(:,:,1,(k-1)*256+n) = single(squeeze(label_im(:,n,:)));
    end
end

% Store results in the imdb struct
imdb.images.data = data ;
imdb.images.label = label ;
%imdb.images.set = set ;

fprintf('\n***** imdb.mat has been created! *****\n');
save('imdbTraining.mat', 'imdb', '-v7.3'); 
