clc
clear
close

%% SVD for a picture
addpath('DATA')
load allFaces.mat

allPersons = zeros(n*6,m*6);
count = 1;
for i=1:6
    for j=1:6
        allPersons(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
            = reshape(faces(:,1+sum(nfaces(1:count-1))),n,m);
        count = count + 1;
    end
end

figure(1), axes('position',[0  0  0 0]), axis off
subplot(1,2,1)
imagesc(allPersons), colormap gray
axis equal

for person = 1:length(nfaces)
    subset = faces(:,1+sum(nfaces(1:person-1)):sum(nfaces(1:person)));
    allFaces = zeros(n*8,m*8);
    
    count = 1;
    for i=1:8
        for j=1:8
            if(count<=nfaces(person)) 
                allFaces(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
                    = reshape(subset(:,count),n,m);
                count = count + 1;
            end
        end
    end
    subplot(1,2,2)
    imagesc(allFaces), colormap gray  
end
axis equal

% size of a single image: 192 x 168
% There are 38 people with each having around 60+ images.

%% SVD for faces

% We use the first 36 people for training data
trainingFaces = faces(:,1:sum(nfaces(1:36)));
avgFace = mean(trainingFaces,2);  % size n*m by 1;

figure
subplot(1,2,1)
imshow(uint8(reshape(avgFace, n, m)))


% compute eigenfaces on mean-subtracted training data
X = trainingFaces-avgFace*ones(1,size(trainingFaces,2));


%%
[U,S,V] = svd(X,'econ');

subplot(1,2,2)
semilogy(diag(S)./sum(diag(S)))
xlim([1 100])

%% Eigen faces

for i=1:50  % plot the first 50 eigenfaces
    pause;  % wait for 0.1 seconds
    imagesc(reshape(U(:,i),n,m)); colormap gray;
    title(num2str(i))
    axis equal
end
