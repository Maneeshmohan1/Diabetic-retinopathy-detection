clear
clc
close all
load('net.mat')
[FileName,PathName] = uigetfile('open the image');
im = (imread([PathName,FileName]));
I2 = imresize(im,[512,512]);
I3 = (imcrop(I2,[50 45 150 120]));
subplot(2,2,1);
imshow(I2)
subplot(2,2,2);
imshow(I3)
I4 = im2bw(I3,.45);
subplot(2,2,3)
imshow(I4)
[j,k]=size(I4);
c=0;
for j=1:120
    for k = 1:150
        if I4(j,k)==0
            c = c+1;
        end
    end
end
if c>19000
    disp('damaged');
    %disp('c----'),disp(c);

    else
    disp('notdamaged');    
    %disp('c----'),disp(c);
    I5=rgb2hsv(I3);
    subplot(2,2,4);
    imshow(I5);
    m=mean2(I5);
    s=std2(I5); 
    p=[c;m;s];
    disp(size(p));
   y=sim(net,p);
    class = vec2ind(y);
if class == 1
    disp('yes');
else
    disp('no');
end

    end