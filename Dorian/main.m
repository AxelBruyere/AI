clear; close all; clc;


%% Détection de droites

%%Lecture de l'image
I = im2double(imread('buildings.png'));

%%Paramètres de l'algorithme
[N1,N2] = size(rgb2gray(I));
%rho
rho_max = sqrt((N1-1)^2+(N2-1)^2);
d_rho = 1/round(max([N1,N2])) * rho_max;
rho = -rho_max : d_rho : rho_max ;
%theta
d_theta = 1/round(max([N1,N2])) * pi;
theta_max = pi - d_theta;
theta = 0 : d_theta : theta_max ;

%%Image des contours
Icont = edge(rgb2gray(I),'Canny');

%%Calcul de la matrice d'accumulation H
H = zeros(length(rho),length(theta));
R = zeros(size(H));
for x=1:N1
    for y=1:N2
        if (Icont(x,y) == 1)
            rau_j = x*cos(theta) + y*sin(theta);
            rau_j = (rau_j + rho_max) * (length(rho)/(2*rho_max));
            rau_j = round(rau_j);
            for i=1:length(theta)
                H(rau_j(i),i) = H(rau_j(i),i) + 1;
            end
        end
    end
end

%%Extraction des maxima locaux
%Seuillage de H
seuil = 0.42 * max(H(:));
H_max = (H >= seuil).*H; 
% H_max = islocalmax(H_max,2); %Matrice des maxima locaux 
                               %(necessite version récente de matlab)
max_ind = find(H_max);
n = length(max_ind);
[Rau,Theta] = ind2sub(size(H),max_ind);
Theta = Theta * d_theta ;
Rau = Rau * d_rho - rho_max;

%%Calcul des points extremaux de chaque droite
P1=zeros(2,n);
P2=zeros(2,n);
for k=1:n
    if Theta(k) == 0 %si pente infinie
        P1(1,k) = Rau(k)/cos(Theta(k));
        P1(2,k) = 1;
        P2(1,k) = Rau(k)/cos(Theta(k));
        P2(2,k) = N1;
    else             %cas general
        P1(1,k) = 1;
        P1(2,k) = Rau(k)/sin(Theta(k));
        P2(1,k) = N2;
        P2(2,k) = (Rau(k)-N2*cos(Theta(k)))/sin(Theta(k));
    end
end

%%Affichage de la matrice H et de ses maxima locaux
figure(1)
imshow(imadjust(H/(max(max(H)))),[], 'XData',theta,...
       'YData',rho,'InitialMagnification','fit');
xlabel('\theta (rad)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)
title(['Matrice d''accumulation H (d\theta = ',num2str(d_theta), ...
        ', d\rho = ', num2str(d_rho),')']);
P = houghpeaks(H,10,'threshold',ceil(0.2*max(H(:))));
x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','blue');

%%Affichage des droites détectées sur l'image d'origine
figure(2)
imshow(I);
title(['Droites détectées avec ',num2str(n),' maxima']);
hold on;
for k=1:n
    plot([P1(1,k),P2(1,k)],[P1(2,k),P2(2,k)]);
end





%% Détection de cercles

%%Image "braille1.png"

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
figure(7)
hold on;
imshow(I);
viscircles(c1,max(r1)*ones(size(r1))-2,'EdgeColor','b');
title('Image d''origine et mise en évidence des points')



%%Image "braille2.jpg"

%Récupération de l'image 
I = im2double(imread('braille2.png'));
I_rgb = rgb2gray(I);

%Debruitage
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

%On trouve les cercles avec la tranformée de Hough
[c1,r1] = imfindcircles(I_dil,[1 40]);

%Affichage
figure(8)
hold on;
imshow(I);
viscircles(c1,max(r1)*ones(size(r1))-2,'EdgeColor','b');
title('Image d''origine et mise en évidence des points')







