clear all; close all; clc;

%%Transformée de Hough

%%Récupération de l'image
I = im2double(imread('buildings.png'));
Icont = edge(rgb2gray(I),'Canny');
[N1,N2] = size(rgb2gray(I));

%%Paramètres
rau_max = sqrt((N1-1)^2+(N2-1)^2);
d_rau = 1/100 * rau_max;
rau = -rau_max : d_rau : rau_max ;

d_theta = 1/200 * pi;
theta_max = pi - d_theta;
theta = 0 : d_theta : theta_max ;

%%Matrice d'accumulation H
H = zeros(length(rau),length(theta));
R = zeros(size(H));

figure(2)
hold on;
for x=1:N1
    for y=1:N2
        if (Icont(x,y) == 1)
            rau_j = x*cos(theta) + y*sin(theta);
            rau_j = (rau_j + rau_max) * (length(rau)/(2*rau_max));
            plot(rau_j)
            rau_j = round(rau_j);
            for i=1:length(theta)
                H(rau_j(i),i) = H(rau_j(i),i) + 1;
            end
        end
    end
end

%%Maxima locaux
seuil = 0.65 * max(max(H));
H_max = (H > seuil).*H;
vec = find(H_max>seuil);
x = [];
y = [];
for i = 1:length(vec)
    [a,b] = ind2sub(size(H_max),vec(i));
    x = [x,a];
    y = [y,b];
end
coord = [x;y];



%%Affichage
figure(1)
subplot(221)
imshow(I);
subplot(222)
imshow(Icont);
subplot(223)
plot(rau_j);
subplot(224)
imshow(H,[min(min(H)) max(max(H))]);

figure(3)
imshow(H_max,[])

figure(4)
imshow(H,[min(min(H)) max(max(H))]);
