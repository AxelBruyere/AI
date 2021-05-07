clear all; close all; clc;

%%Detection de cercles bis

I1 = im2double(imread('braille1.png'));
I2 = im2double(imread('braille2.png'));

I1_rgb = rgb2gray(I1);
I2_rgb = rgb2gray(I2);

[H1,L1] = size(rgb2gray(I1));
[H2,L2] = size(rgb2gray(I2));

% %%Lissage de I1_rgb
% R = 3;
% SE = strel('disk', R, 0);
% I1_rgb = imerode(imdilate(imdilate(imerode(I1_rgb,SE),SE),SE),SE);

%%Top-Hat ouvert de I1_rgb
R1 = 3;
S1 = strel('disk', R1, 0);
I1_ouv = imdilate(imerode(I1_rgb,S1),S1);

I1_th = I1_rgb - I1_ouv;
% I1_th = im2bw(I1_th,graythresh(I1_th));

R2=3;
S2 = strel('disk', R2, 0);
I1_th_ouv = imdilate(I1_th,S2);
figure(1)
subplot(221)
imshow(I1_rgb);
subplot(222)
imshow(imgaussfilt(I1_ouv));

subplot(223)
imshow(I1_th,[min(min(I1_th)) max(max(I1_th))]);
subplot(224)
imshow(I1_th_ouv,[min(min(I1_th_ouv)) max(max(I1_th_ouv))]);
