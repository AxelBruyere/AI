clear all; close all; clc;

%%Detection de cercles

I1 = im2double(imread('braille1.png'));
I2 = im2double(imread('braille2.png'));

I1_rgb = rgb2gray(I1);
I2_rgb = rgb2gray(I2);

I1_cont = edge(rgb2gray(I1),'Canny');
I2_cont = edge(rgb2gray(I1),'Canny');

[H1,L1] = size(rgb2gray(I1));
[H2,L2] = size(rgb2gray(I2));

%%Debruitage
[Gmag1, Gdir1] = imgradient(I1_rgb,'prewitt');

R1 = 2;
SE_e = strel('disk', 2, 0);
SE_d = strel('disk', 5, 0);
SE_10 = strel('disk', 20, 0);

I1_liss = imerode(imdilate(imdilate(imerode(I1_rgb,SE_e),SE_d),SE_d),SE_e);
I1_liss_th = I1_liss - imdilate(imerode(I1_liss,SE_10),SE_10);

%[Gmag2, Gdir2] = imgradient(I1_liss,'prewitt');



%%Affichage
figure(1)
subplot(221)
imshow(I1)
subplot(222)
imshow(I1_liss);
subplot(223)
imshow(Gmag1,[0 1])
subplot(224)
imshow(I1_liss_th,[0 1])


%%circles functions

SEd = strel('disk', 2, 0);
I = imdilate(Gmag1,SEd);
% I = Gmag1;
% I = im2bw(I, graythresh(I));

figure(2)
imshow(I1);
hold on;

[c1,r1] = imfindcircles(I1_liss_th,[4 40]);
% [c2,r2] = imfindcircles(I2,[3 50]);
viscircles(c1,max(r1)*ones(size(r1))-2,'EdgeColor','b');