clear; close all; clc;

%% Sélection de l'exercice
% 1 -> Détections de droites
% 2 -> Détections de cercles
exercice = 1;
switch exercice
%% Détection de droites
    case 1
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
