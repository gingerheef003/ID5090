clc
clear
close all

m = 6;
n = 10;
N = m*n;

ZOR = 1;
ZOO = 10 + ZOR;
ZOA = 10 + ZOO;

t_steps = 3*1e3;
dt = 0.1;

sight = 270;
turn_rate = pi/8;
err = 0.0;

L = 50; % Domain size
tankcenterX = 0;
tankcenterY = 0;

pos = zeros(N,2,t_steps);
vel = zeros(N,2,t_steps);

% Initial configuration (square arrangement)
pos(:,1,1) = (3*rem((0:N-1),n))';
pos(:,2,1) = (3*floor((0:N-1)/n))';
pos(:,1,1) = pos(:,1,1) - mean(pos(:,1,1)) + tankcenterX;
pos(:,2,1) = pos(:,2,1) - mean(pos(:,2,1)) + tankcenterY;

% Initial headings
th0 = rand(N,1) * 2*pi;
vel(:,:,1) = [cos(th0) sin(th0)];

% boundary
thc = 0:0.1:2*pi; xc = L*cos(thc); yc = L*sin(thc); plot(xc,yc, '--k', LineWidth=3); hold on
% arrows for agents
quiver(pos(:,1,1), pos(:,2,1), vel(:,1,1), vel(:,2,1), "off", "Marker", ".", "ShowArrowHead", "on", "MarkerSize",10);
axis equal;
axis(L*[-1 1 -1 1])
axis off
hold off

for t = 2:t_steps
    vel(:,:,t) = vel(:,:,t-1);

    dist_ij = squareform(pdist(pos(:,:,t-1)));
    isreplusion = dist_ij <= ZOR;
    for k=1:N
        isreplusion(k,k) = 0;
    end

    isalignment = dist_ij>ZOR & dist_ij<=ZOO;
    isattraction = dist_ij>ZOO & dist_ij<=ZOA;

    issight = zeros(N);
    for j = 1:N
        for js = 1:N
            if j~=js
                angle = acosd(dot(vel(j,:,t-1), diff(pos([js,j],:,t-1),1)./norm(diff(pos([js,j],:,t-1),1))));
                if angle < sight
                    issight(j,js) = 1;
                end
            end
        end
    end

    isalignment = isalignment .* issight;
    isattraction = isattraction .* issight;

    for j = 1:N
        if sqrt((pos(j,1,t-1) - tankcenterX).^2 + (pos(j,2,t-1) - tankcenterY).^2) >= L
            db(1) = (tankcenterX - pos(j,1,t-1));
            db(2) = (tankcenterY - pos(j,2,t-1));
            db = db./vecnorm(db);

            dberror = pi/8 * randn(1,2);
            db = db + dberror;

            db = db./vecnorm(db);
            dj = db;
        elseif sum(isreplusion(j,:)) > 0
            jj = find(isreplusion(j,:));

            dr = [mean(pos(j,1,t-1) - pos(jj,1,t-1)) mean(pos(j,2,t-1) - pos(jj,2,t-1))];
            dr = dr./vecnorm(dr);

            dj = dr;
        elseif sum(isalignment(j,:))>0 || sum(isattraction(j,:))>0
            jk = find(isalignment(j,:));
            jl = find(isattraction(j,:));

            if numel(jk)>0 && numel(jl)>0
                dal = [mean(vel(jk,1,t-1)) mean(vel(jk,2,t-1))];
                dal = dal./vecnorm(dal);

                datt = [mean(pos(jl,1,t-1)) mean(pos(jl,2,t-1))];
                datt = datt./vecnorm(datt);

                ds = (dal + datt)/2;
            elseif numel(jk)>0
                dal = [mean(vel(jk,1,t-1)) mean(vel(jk,2,t-1))];
                dal = dal./vecnorm(dal);

                ds = dal;
            elseif numel(jl)>0
                datt = [mean(pos(jl,1,t-1)) mean(pos(jl,2,t-1))];
                datt = datt./vecnorm(datt);

                ds = datt;
            end
            dj = ds;
        end

        dj = dj + err*rand(1,2);
        dj = dj./norm(dj);

        turn_angle = atan2d((dj(2)*vel(j,1,t)-dj(1)*vel(j,2,t)), dj(1)*dj(2) + vel(j,1,t)*vel(j,2,t));

        if abs(turn_angle) > turn_rate*dt
            thetaj = atan2(vel(j,2,t), vel(j,1,t)) + sign(turn_angle)*turn_rate*dt;
            vel(j,:,t) = [cos(thetaj) sin(thetaj)];
        else
            vel(j,:,t) = dj;
        end
    end


    pos(:,:,t) = pos(:,:,t-1) + vel(:,:,t)*dt;

    thc = linspace(0,2*pi,50); xc = L*cos(thc); yc = L*sin(thc); plot(xc,yc, '--k', LineWidth=3); hold on
    quiver(pos(:,1,t), pos(:,2,t), vel(:,1,t), vel(:,2,t), "off", "Marker", ".", "ShowArrowHead", "on", "MarkerSize",10);

    axis equal;
    axis(L*[-1 1 -1 1])
    axis off
    hold off

    drawnow limitrate % make it seem like a movie
end