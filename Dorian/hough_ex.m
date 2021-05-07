clear; close all; clc;

%%Lecture image
I = imread('buildings.png');
rotI = imrotate(I,0,'crop');
%rotI = imgaussfilt(I);

%%Detection des bords
BW = edge(rgb2gray(rotI),'canny');


%%Affichage
figure(1)
subplot(121)
imshow(rotI)
subplot(122)
imshow(BW)

%%Transformee de Hough
[H,theta,rho] = hough(BW);

figure(2)
imshow(imadjust(H/(max(max(H)))),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)

%%Maxima
P = houghpeaks(H,10,'threshold',ceil(0.2*max(H(:))));
x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','black');

%%Lignes
lines = houghlines(BW,theta,rho,P,'FillGap',5,'MinLength',7);
figure(3), imshow(rotI), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');

