% This file contains the simulation code of Figure 2 (Spectral Efficiency VS Power)
% of the paper titled "Low-Complexity Sum-Capacity Maximization for Intelligent
% Reflecting Surface-Aided MIMO Systems," accepted by the IEEE Wireless
% Communications Letters.

% In Fig. 2, we provide a plot for the exhaustive search plot. Due to some
% reasons, the plot is not provided here. But if the reader wants to perform
% the exhaustive search method, we provide the function via the revoke of
% DsmMimo.main_exh().

% Creator:
%   Ahmad Sirojuddin
%   sirojuddin.p@gmail.com

% This file dependencies:
%   DsmMimo.m
%   irsMimo.m

clear;
clc;

chSampleNum = 10;               % number of channel samples
P_set = [-10 -5 0 5 10 15 20]'; % Set of powers to be plot, in dBm
sig_n = irsMimo.dBm_to_watt(0); % noise power

K = 16; % Number of Transmit Antenna
Nt = K; % Nt is an alias for K in [7]
L = 12; % Number of Receive Antenna
Nb = L; % Nb is an alias for L in [7]
N_set = [64 16]'; % The set of number of reflecting elements
alpha = [0.053 0.13]'; % step size for the gradient ascent method

%------- Parameters Related to Channel Generation -------%
ricianFact = irsMimo.dB_to_linear(10); % 10 dB = 10
dist_Hd = 1; % direct link distance
dist_M = 1; % S-IRS distance
dist_Hr = 1; % IRS-D distance
pathLossRef = irsMimo.dB_to_linear(0); % pathloss at the reference distance
distRef = 1; % reference distance
pathLossExp_freeSpc = 2; % pathlos exponent for channels G and H
pathLossExp_urb = 3; % pathlos exponent for channel F

%------- Antenna Tx & Rx Response (see [7]) -------%
freq = 2.4e9;
phi = pi/10;

iterLimit = 1000; % maximum iteration number
tol = 1e-5; % tolerance

beta = 1; % a parameter in [7]

%------- Rate of all methods to be plotted, initialization -------%
rateDsm = zeros(size(N_set,1), size(P_set,1));
rateAdmm = zeros(size(N_set,1), size(P_set,1));
rateGA = zeros(size(N_set,1), size(P_set,1));

for N_cnt = 1:length(N_set)
    N = N_set(N_cnt); % number of reflecting elements 
    Nr = N; % Alias for N in [7]
    
    rateDsm_temp = zeros(chSampleNum, size(P_set,1)); % Initialization
    rateAdmm_temp = zeros(chSampleNum, size(P_set,1)); % Initialization
    rateGA_temp = zeros(chSampleNum, size(P_set,1)); % Initialization
    for chSampleCnt = 1:chSampleNum
        disp(['N = ', num2str(N), '; chSampleCnt = ', num2str(chSampleCnt), ' -------']);
        %------- Generating Channels -------%
        Hd = irsMimo.ricianCh([Nt,Nb], ricianFact, dist_Hd, pathLossRef, distRef, pathLossExp_urb, freq, phi);
        Hr = irsMimo.ricianCh([Nr,Nb], ricianFact, dist_Hr, sqrt(pathLossRef), distRef, pathLossExp_freeSpc, freq, phi);
        M = irsMimo.ricianCh([Nr,Nt], ricianFact, dist_M, sqrt(pathLossRef), distRef, pathLossExp_freeSpc, freq, phi);
        
        H=M; G=Hr'; F=Hd'; % Our Channels notation relating to [7]
        init_theta = 2*pi*(rand(N,1)-0.5);
        for P_cnt = 1:length(P_set)
            P = irsMimo.dBm_to_watt(P_set(P_cnt)); disp(['   P : ', num2str(P), ' -------']);

            %------- Calculating rate of all considered methods -------%
            rateDsm_temp(chSampleCnt, P_cnt) = DsmMimo.main(G, H, F, P, sig_n, init_theta, iterLimit); disp(['      rateDsm = ', num2str(rateDsm_temp(chSampleCnt, P_cnt))]);
            rateAdmm_temp(chSampleCnt, P_cnt) = irsMimo.main(P, beta, sig_n, Hd, M, Hr, iterLimit, init_theta, tol); disp(['      rateAdmm = ', num2str(rateAdmm_temp(chSampleCnt, P_cnt))]);
            rateGA_temp(chSampleCnt, P_cnt) = DsmMimo.main_GA1(G, H, F, P, sig_n, init_theta, alpha(N_cnt), iterLimit); disp(['      rateGA1 = ', num2str(rateGA_temp(chSampleCnt, P_cnt))]);
        end
    end
    %------- Averaging rate over channel realization number -------%
    rateDsm(N_cnt, :) = mean(rateDsm_temp, 1);
    rateAdmm(N_cnt, :) = mean(rateAdmm_temp, 1);
    rateGA(N_cnt, :) = mean(rateGA_temp, 1);
end
lineColors = {[0 0 150]/255, [0 130 0]/255, [210 0 0]/255, [102 51 0]/255, [50 50 0]/255, [0 50 50]/255, [45 14 14]/255};
lineStyles = {'-', '--', ':', '-.'};
lineMarkers = {'o', '+', 'square', 'x', '*', 'd', '^', 'v', '>', '<'};
figure('Name', 'SE vs Power');

for N_cnt=1:size(N_set,1)
    p = plot(P_set', rateDsm(N_cnt,:));
    p.Color = lineColors{1};
    p.Marker = lineMarkers{1};
    p.MarkerSize = 11;
    p.LineStyle = lineStyles{N_cnt};
    p.LineWidth = 2;
    hold on;
    
    p = plot(P_set', rateAdmm(N_cnt,:));
    p.Color = lineColors{2};
    p.Marker = lineMarkers{2};
    p.MarkerSize = 11;
    p.LineStyle = lineStyles{N_cnt};
    p.LineWidth = 2;
    hold on;
    
    p = plot(P_set', rateGA(N_cnt,:));
    p.Color = lineColors{3};
    p.Marker = lineMarkers{3};
    p.MarkerSize = 11;
    p.LineStyle = lineStyles{N_cnt};
    p.LineWidth = 2;
    hold on;
end
xlabel('Source Power (dBm)');
ylabel('SE (bps/Hz)');
% This is a temporary legend. We use graphic design software to design
% the legend and the other figure properties.
legend('DSM', 'ADMM', 'GA');