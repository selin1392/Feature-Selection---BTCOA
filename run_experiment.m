clc;
clear;
close all;

% Add paths
addpath(genpath('src'));
addpath(genpath('preprocessing'));
addpath(genpath('evaluation'));

% Fix random seed for reproducibility
rng(2024, 'twister');

% Main execution
MainCodeFS;
