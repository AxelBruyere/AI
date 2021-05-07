clear all; close all; clc;

%%Detection de cercles
I1 = im2double(imread('braille1.png'));
I1_rgb = rgb2gray(I1);

%%Debruitage

%Top hat avec un disque de 15 pixels comme élément structurant
se = strel('disk',14);
J = imtophat(I1_rgb,se).*255;
%Seuillage
BW = (J>18).*J;

%On trouve les cercles avec la tranformée de Hough
[c1,r1] = imfindcircles(BW,[3 40]);

%%Affichage
figure(1)
subplot 121
imshow(I1)
subplot 122
hold on;
imshow(I1);
viscircles(c1,max(r1)*ones(size(r1))-2,'EdgeColor','b');