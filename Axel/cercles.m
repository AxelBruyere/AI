clear all; close all; clc;

%%Récupération de l'image
I = im2double(imread('braille1.png'));
I_rgb = rgb2gray(I);

%%Debruitage
%Top hat avec un disque de 8 pixels comme élément structurant
se = strel('disk',8);
I_th = imtophat(I_rgb,se).*255;
%Seuillage
th_seuil = (I_th>9).*I_th;

%%Recherche des cercles
[c1,r1] = imfindcircles(th_seuil,[4 40]);

%%Affichage
figure(1)
hold on;
imshow(I);
viscircles(c1,max(r1)*ones(size(r1))-2,'EdgeColor','b');
sgtitle('Image d''origine et mise en évidence des points')