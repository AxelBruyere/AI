clear all; close all; clc;

%%Detection de cercles bis

I = im2double(imread('braille2.png'));
I_rgb = 1 - rgb2gray(I);
[H1,L1] = size(rgb2gray(I));

NHOOD1 = [0 0 0; 0 1 1; 0 1 0];
S1 = strel('arbitrary', NHOOD1);
I_ouv = imdilate(imerode(I_rgb,S1),S1);

R2=3;
S2 = strel('disk', R2, 0);
I_ferm = imerode(imdilate(I_ouv,S2),S1);


figure(1)
subplot(221)
imshow(I_rgb);
hold on;
[c2,r2] = imfindcircles(I_ferm,[2 6]);
viscircles(c2,max(r2)*ones(size(r2)),'EdgeColor','b');
subplot(222)
imshow(I_ouv);
subplot(223)
imshow(I_ferm);
subplot(224)
imshow(I_ferm);
hold on;
[c1,r1] = imfindcircles(I_ferm,[2 6]);
viscircles(c1,max(r1)*ones(size(r1)),'EdgeColor','b');