clear all; close all; clc;

%%Récupération de l'image 
I = im2double(imread('braille2.png'));
I_rgb = rgb2gray(I);



%%Debruitage
filtre = fspecial('gaussian',[50,50],0.1);
I_filtre = (1-imfilter(I_rgb,filtre)).*255;
%Seuillage
I_seuil1 = (I_filtre>17).*I_filtre;
I_seuil2 = (I_seuil1>18).*255;
%Morphologie mathématique
S = [0 0 0; 0 1 0; 0 1 0];
I_ouv1 = imopen(I_seuil2,S);
S2 = [1,1];
I_ouv2 = imopen(I_ouv1,S2);
I_dil = imdilate(I_ouv2,ones(4));


%%On trouve les cercles avec la tranformée de Hough
[c1,r1] = imfindcircles(I_dil,[1 40]);

%%Affichage
figure(8)
hold on;
imshow(I);
viscircles(c1,max(r1)*ones(size(r1))-2,'EdgeColor','b');
sgtitle('Image d''origine et mise en évidence des points')