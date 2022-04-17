% This file contains the simulation code of Figure 4 (Execution Time VS N)
% of the paper titled "Low-Complexity Sum-Capacity Maximization for Intelligent
% Reflecting Surface-Aided MIMO Systems," accepted by the IEEE Wireless
% Communications Letters.

% In Fig. 2, we provide a plot for the exhaustive search plot. Due to some
% reasons, the plot is not provided here.

% Creator:
%   Ahmad Sirojuddin
%   sirojuddin.p@gmail.com

% This file dependencies:
%   DsmMimo.m
%   irsMimo.m

clear;
clc;

chSampleNum = 50; % number of channel samples
P_set = 20; % Tx power, 20 dBm
P = irsMimo.dBm_to_watt(P_set); % convert P_set to watt
K = 16; % Number of Transmit Antenna
Nt = K; % Nt is an alias for K in [7]
L = 12; % Number of Receive Antenna
Nb = L; % Nb is an alias for L in [7]
N_set = [5 10 15 20 25 30 35 40]'; % The set of number of reflecting elements
alpha = [0.2 0.16 0.13 0.11 0.1 0.09 0.08 0.076]'; % step size for the gradient ascent method
tol_set = [1e-2 1e-3]'; % set of tolerance values
iterLimit = 1000; % maximum iteration number

%------- Parameters Related to Channel Generation -------%
ricianFactor = irsMimo.dB_to_linear(10); % 10 dB = 10
dist = 1; % distance for all channels (we normalize them)
pathLossRef = irsMimo.dB_to_linear(0); % pathloss at the reference distance
distRef = 1; % reference distance
pathLossExp = 2; % path-loss exponent
freq = 2.4e9;
phi = pi/10;

beta = 1; % a parameter in [7]
sig_n = irsMimo.dBm_to_watt(0); % noise power

%------- Time computation of all methods to be plotted, initialization-------%
timeDsm = zeros(length(tol_set), length(N_set));
timeAdmm = zeros(length(tol_set), length(N_set));
timeGA = zeros(length(tol_set), length(N_set));
for n = 1:length(N_set)
    N = N_set(n); % number of reflecting elements
    Nr = N; % Alias for N in [7]

    timeDsm_temp = zeros(length(tol_set), chSampleNum); % Initialization
    timeAdmm_temp = zeros(length(tol_set), chSampleNum); % Initialization
    timeGA_temp = zeros(length(tol_set), chSampleNum); % Initialization
    for chSampleCnt = 1:chSampleNum
        clc;
        disp(['N = ', num2str(N), '; progress = ', num2str(chSampleCnt/chSampleNum*100), ' percent']);

        %------- Generating Channels -------%
        Hd = irsMimo.ricianCh([Nt,Nb], ricianFactor, dist, pathLossRef, distRef, pathLossExp, freq, phi);
        Hr = irsMimo.ricianCh([Nr,Nb], ricianFactor, dist, pathLossRef, distRef, pathLossExp, freq, phi);
        M = irsMimo.ricianCh([Nr,Nt], ricianFactor, dist, pathLossRef, distRef, pathLossExp, freq, phi);

        H=M; G=Hr'; F=Hd'; % Our Channels notation relating to [7]
        init_theta = 2*pi*(rand(N, 1)-0.5);
        for tol_th = 1:length(tol_set)
            tol = tol_set(tol_th);

            %------- Calculating the iteration numbers of all considered methods -------%
            [~, ~, ~, ~, timeDsm_temp(tol_th, chSampleCnt)] = DsmMimo.maximize_psi_dsm(G, H, F, init_theta, iterLimit, tol);
            [~, ~, ~, ~, timeGA_temp(tol_th, chSampleCnt)] = DsmMimo.maximize_psi_GA1(G, H, F, init_theta, alpha(n), iterLimit, tol);
            [~, ~, ~, ~, timeAdmm_temp(tol_th, chSampleCnt)] = irsMimo.main(P, beta, sig_n, Hd, M, Hr, iterLimit, init_theta, tol);
            timeAdmm_temp(tol_th, chSampleCnt) = timeAdmm_temp(tol_th, chSampleCnt)/1000;
            % NOTE: TimeAdmm_temp is in millisecond in our machine, 
            % while the others are in second. Therefore, we divide it by 1000.
        end
    end
    timeDsm(:, n) = mean(timeDsm_temp, 2);
    timeAdmm(:, n) = mean(timeAdmm_temp, 2);
    timeGA(:, n) = mean(timeGA_temp, 2);
end

lineColors = {[0 0 150]/255, [0 130 0]/255, [210 0 0]/255, [102 51 0]/255, [50 50 0]/255, [0 50 50]/255, [45 14 14]/255};
lineStyles = {'-', '--', ':', '-.'};
lineMarkers = {'o', '+', 'x', 'square', '*', 'd', '^', 'v', '>', '<'};
figure('Name', 'Time vs N');

for tol_th = 1:size(tol_set,1)
    %--- DSM ---%
    p = plot(N_set', timeDsm(tol_th, :));
    p.Color = lineColors{1};
    p.Marker = lineMarkers{1};
    p.MarkerSize = 11;
    p.LineWidth = 2;
    p.LineStyle = lineStyles{tol_th};
    hold on;
    
    %--- ADMM ---%
    p = plot(N_set', timeAdmm(tol_th, :));
    p.Color = lineColors{2};
    p.Marker = lineMarkers{2};
    p.MarkerSize = 11;
    p.LineWidth = 2;
    p.LineStyle = lineStyles{tol_th};
    hold on;
    
    %--- GA1 ---%
    p = plot(N_set', timeGA(tol_th, :));
    p.Color = lineColors{3};
    p.Marker = lineMarkers{3};
    p.MarkerSize = 11;
    p.LineWidth = 2;
    p.LineStyle = lineStyles{tol_th};
    hold on;
end
xlabel('Number of IRS(N)');
ylabel('Execution Time (s)');
% This is a temporary legend. We use graphic design software to design
% the legend and the other figure properties.
legend('DSM', 'ADMM', 'GA');