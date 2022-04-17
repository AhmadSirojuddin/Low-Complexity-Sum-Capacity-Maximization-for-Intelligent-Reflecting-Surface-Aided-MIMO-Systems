classdef irsMimo < handle
    methods(Static)
        function [rate, v, F, objTrace, nIter, timeExec] = main(P, beta, sig_n, Hd, M, Hr, iterLimit, init_theta, tol)
            irsMimo.filter(P, beta, sig_n, Hd, M, Hr);
            Nb = size(Hr', 1); Nr = size(M, 1);
            
            K = M*Hd;
            [T_hat, largestEig] = irsMimo.get_T_hat(beta, Hr, M, K, Nb, Nr);
            
            %--- ADMM ---%
            tic;
            rho = max(sqrt(2*largestEig), largestEig);
            x_init = exp(1j*[init_theta; 0]);
            nu_init = randn(Nr+1, 1) + 1j*randn(Nr+1,1);
            [~, x_opt, ~, objTrace, nIter, ~] = irsMimo.admm(T_hat, rho, x_init, nu_init, iterLimit, tol, Hr, M, Hd);
            
            v = x_opt(1:end-1)./x_opt(end);
            timeExec = toc;
            H_eff = (beta*Hr'*diag(v)*M + Hd')';
            
            %--- SVD and Water Filling ---%
            Ns = rank(H_eff);
            [~, Lambda, ~] = svd(H_eff');
            lambda = diag(Lambda);
            heightIn = sig_n*Ns./(P*lambda.^2);
            [pow_alloc, ~] = irsMimo.waterFilling(ones(size(heightIn)), heightIn, Ns);
            F = 0;
            rate = irsMimo.rate_tilde(P, pow_alloc, lambda, sig_n, Ns);
            
        end
        function filter(P, beta, sig_n, Hd, M, Hr)
            if ~isreal(P) || ~isscalar(P) || P<0
                disp('ERROR! the input P must be a real scalar positive');
                disp('your input P :'); disp(P);
                error(' ');
            end
            if ~isreal(beta) || ~isscalar(beta) || beta<0
                disp('ERROR! the input beta must be a real scalar positive');
                disp('your input beta :'); disp(beta);
                error(' ');
            end
            if ~isreal(sig_n) || ~isscalar(sig_n) || sig_n<0
                disp('ERROR! the input sig_n must be a real scalar positive');
                disp('your input sig_n :'); disp(sig_n);
                error(' ');
            end
            if ~isequal(size(Hd', 1), size(Hr', 1)) %Nb
                disp('ERROR! Hd column size and Hr column size must be the same');
                error(' ');
            end
            if ~isequal(size(Hd', 2), size(M, 2)) %Nt
                disp('ERROR! Hd row size and M column size must be the same');
                error(' ');
            end
            if ~isequal(size(M, 1), size(Hr', 2)) %Nr
                disp('ERROR! Hd row size and M column size must be the same');
                error(' ');
            end
        end
        function [T_hat, largestEig] = get_T_hat(beta, Hr, M, K, Nb, Nr)
            T_11 = 0;
            for i=1:Nb
                T_11 = T_11 + diag(Hr(:,i)')*(M*M')*diag(Hr(:,i));
            end
            T_11 = -(beta^2)*T_11;
            
            T_12 = 0;
            for i=1:Nb
                T_12 = T_12 + diag(Hr(:,i)')*K(:,i);
            end
            T_12 = -beta * T_12;
            T = [T_11, T_12; T_12' 0];
            eig_T = sort(real(eig(T)));
            T_hat = T - eig_T(1)*eye(Nr+1);
            eig_T_hat = sort(real(eig(T_hat)));
            if real(eig_T_hat) < 0
                disp('ERROR! T_hat is not semidefinite positive');
                error(' ');
            else
                largestEig = eig_T_hat(end);
            end
        end
        function out = objective_11a(T_hat, x)
            out = 0.5*x'*T_hat*x;
        end
        function [out, out2] = objective_8(Hr, M, Hd, Theta)
            out = abs(trace((Hr'*Theta*M + Hd')*(Hr'*Theta*M + Hd')'));
            out2 = (norm(Hd' + Hr'*Theta*M, 'fro'))^2;
        end
        function [v, Theta] = get_v_Theta(u)
            v = u(1:end-1)./u(end);
            Theta = diag(conj(v));
        end
        function [u_now, x_now, nu_now, objTrace, k, psiTrace] = admm(T_hat, rho, x_prev, nu_prev, iterLimit, tol, Hr, M, Hd)
            k = 0;
            objTrace = zeros(iterLimit, 1);
            objPrev = irsMimo.objective_11a(T_hat, x_prev);
            objTrace(1) = irsMimo.objective_11a(T_hat, x_prev);
            err = inf;
            psiTrace = zeros(iterLimit, 1);
            [~, Theta] = irsMimo.get_v_Theta(exp(1j*angle(x_prev)));
            psiTrace(1,1) = irsMimo.objective_8(Hr, M, Hd, Theta);
            while (k<iterLimit && err>tol)
                k = k+1;
                
                % The following commented statements evaluate the values
                % of u and x in each iteration update to ensure that
                % our code is true.
                u_now = exp(1j*(angle(rho*x_prev - nu_prev))); %disp('   u_now :'); disp(u_now.');
                %dL_dtheta = irsMimo.dL_dtheta(x_prev, u_now, nu_prev, rho); disp('   dL_dtheta (CHECK = 0???):'); disp(dL_dtheta.');
                %obj_v1 = irsMimo.Lagrangian_12(T_hat, x_prev, u_now, nu_prev, rho); disp(['   obj_v1 (Lag 12): ', num2str(obj_v1)]);
                %d2L_dtheta2 = irsMimo.d2L_dtheta2(x_prev, u_now, nu_prev, rho); disp('   d2L_dtheta2 (convexity) :'); disp(d2L_dtheta2.');
                
                x_now = (rho*eye(size(T_hat))+T_hat) \ (nu_prev + rho*u_now); %disp(' '); disp('   x_now :'); disp(x_now.');
                %disp(abs(x_now.'));
                %dL_dx = irsMimo.dL_dx(T_hat, x_now, u_now, nu_prev, rho); disp('   dL_dx(CHECK = 0???) :'); disp(dL_dx.');
                
                nu_now = T_hat*x_now;
                
                objNow = irsMimo.objective_11a(T_hat, x_now);
                objTrace(k+1) = irsMimo.objective_11a(T_hat, x_now); %may use u_now instead
                err = abs(objNow - objPrev);
                
                x_prev = x_now;
                nu_prev = nu_now;
                objPrev = objNow;
                
                % The following statements record the value of the objective
                % in (8) in each iteration update. They can be commented out.
                [~, Theta] = irsMimo.get_v_Theta(exp(1j*angle(x_now)));
                [psiTrace(k+1,1), ~] = irsMimo.objective_8(Hr, M, Hd, Theta);
            end
            psiTrace(k+2:end)=[];
        end
        
        %------- Water Filling Algorithm -------%
        % The following water filling code uses the concept of pouring water
        % into a set of stairs with different levels. We develop a better
        % water-filling algorithm by using the line search method
        % (the Newton-Raphson method to be more specific)
        % in the following link:
        % https://github.com/JuddWirelessComm/convex_optimization/tree/main/water_filling
        function [allocReturn, waterLvl] = waterFilling(lengthIn, heightIn, totalVolume)
            if ~isvector(lengthIn) || ~isvector(heightIn)
                disp('ERROR! parameter lengthIn and heightIn must be a vector');
                disp('lengthIn :'); disp(lengthIn); disp('heightIn :'); disp(heightIn);
                error(' ');
            end
            if ~isequal(size(lengthIn), size(heightIn))
                disp('ERROR! parameter lengthIn and heightIn must be the same size');
                error(' ');
            end
            if ~isreal(lengthIn) || ~isreal(heightIn)
                disp('ERROR! parameter lengthIn and heightIn must be real');
                error(' ');
            end
            if ~isscalar(totalVolume) || totalVolume<0
                disp('ERROR! totalVolume should be a real scalar');
                error(' ');
            end
            %-------Sort the data first-------%
            [height, index] = sort(heightIn);
            len = lengthIn(index);
            stairsNbr = size(len, 1);

            %-------Initialization-------%
            remnantWater = totalVolume;
            waterLvl = height(1);

            %-------Finding Water Level-------%
            roomVol = zeros(stairsNbr, 1);
            for stairsCnt=1:stairsNbr
                if stairsCnt<stairsNbr
                    roomVol(stairsCnt) = sum(len(1:stairsCnt))*(height(stairsCnt+1)-height(stairsCnt));
                end

                if (roomVol(stairsCnt)<remnantWater && stairsCnt<stairsNbr)
                    remnantWater = remnantWater - roomVol(stairsCnt);
                    waterLvl = waterLvl + (height(stairsCnt+1)-height(stairsCnt));
                else        
                    roomVol(stairsCnt) = remnantWater;
                    waterLvl = waterLvl + roomVol(stairsCnt)/sum(len(1:stairsCnt));
                    break;
                end
            end

            %-------Calculating Water volume above each stairs-------%
            alloc = zeros(stairsNbr, 1);
            allocReturn = zeros(stairsNbr, 1);
            wateredStairs = stairsCnt;
            for stairsCnt=1:wateredStairs
                alloc(stairsCnt) = len(stairsCnt)*(waterLvl-height(stairsCnt));
            end

            %-------Sorting Back to Original Order-------%
            for stairsCnt=1:stairsNbr
                allocReturn(index(stairsCnt)) = alloc(stairsCnt);
            end
        end
        function out = rate(P, sig_n, Ns, H_eff, F)
            mat = P/(sig_n*Ns)*(H_eff'*(F*F')*H_eff);
            out = log2(det(eye(size(mat)) + mat));
        end
        function vectOut = lengthenTrimVector(vectIn, len, tol)
            if size(vectIn, 2)~= 1
                error('The parameter "vectIn" should be a column vector');
            elseif ~isreal(vectIn)
                error('The parameter "vectIn" should be a real vector');
            end
            if nargin==3
                for vectIdx = 2:size(vectIn, 1)
                    len = vectIdx;
                    if ((vectIn(vectIdx, 1)-vectIn(vectIdx-1, 1)) < tol)
                        break;
                    end
                end
            end
            vectOut = vectIn;
            if size(vectIn,1) > len
                vectOut(len+1:size(vectIn,1)) = [];                              %Trim the vector
            else
                vectOut = [vectOut; vectOut(end)*ones(len-size(vectOut,1), 1)];  %lengthen the vector
            end
        end
        
        %----------- Functions for Verifying -----------%
        function out = dL_dx(T_hat, x, u, nu, rho)
            out = 0.5*(T_hat*x - nu - rho*(u-x));
        end
        function out = dL_dtheta(x, u, nu, rho)
            ej_theta = exp(1j*angle(u));
            out = imag(diag(ej_theta)*(rho*conj(x)-conj(nu)));
        end
        function out = d2L_dtheta2(x, u, nu, rho)
            ej_theta = exp(1j*angle(u));
            out = imag(diag(ej_theta)*(rho*conj(x)-conj(nu)));
        end
        function out = Lagrangian_12(T_hat, x, u, nu, rho)
            out = 0.5*x'*T_hat*x + real(nu'*(u-x)) + 0.5*rho*norm(u-x)^2;
        end
        function out = rate_tilde(P, power_alloc, lambda, sig_n, Ns)
            if ~isreal(P) || ~isscalar(P) || P<0
                disp('ERROR! input P must be a real positive');
                error(' ');
            end
            if ~isreal(sig_n) || ~isscalar(sig_n) || sig_n<0
                disp('ERROR! input P must be a real positive');
                error(' ');
            end
            if ~isreal(Ns) || ~isscalar(Ns) || Ns<0
                disp('ERROR! input P must be a real positive');
                error(' ');
            end
            if ~isequal(size(power_alloc), size(lambda))
                disp('ERROR! input power_alloc and lambda must be the same size');
                error(' ');
            end
            if ~iscolumn(power_alloc)
                disp('ERROR! input power_alloc must be a column vector');
                error(' ');
            end
            out = sum(log2(1+P.*power_alloc.*(lambda.^2)./(sig_n*Ns)));
        end
        
        %----------- Functions to be revoked in other files -----------%
        function ch = ricianCh(chSize, ricianFactor, dist, pathLossRef, distRef, pathLossExp, freq, phi)
            if ~isequal(size(chSize), [1 2]) || ~isreal(chSize)
                disp('ERROR! input chSize must be in size of 2x1 and real');
                disp('your chSize : '); disp(chSize);
                error(' ');
            end
            if ~isreal(ricianFactor) || ~isscalar(ricianFactor) || ricianFactor<0
                disp('ERROR! input kappa must be a real scalar positive');
                disp('your kappa :'); disp(ricianFactor);
                error(' ');
            end
            if ~isreal(dist) || ~isscalar(dist) || dist<0
                disp('ERROR! input dist must be a real scalar positive');
                disp('your dist :'); disp(dist);
                error(' ');
            end
            if ~isreal(pathLossRef) || ~isscalar(pathLossRef) || pathLossRef<0
                disp('ERROR! input pathLossRef must be a real scalar positive');
                disp('your pathLossRef :'); disp(pathLossRef);
                error(' ');
            end
            if ~isreal(distRef) || ~isscalar(distRef) || distRef<0
                disp('ERROR! input distRef must be a real scalar positive');
                disp('your distRef :'); disp(distRef);
                error(' ');
            end
            if ~isreal(pathLossExp) || ~isscalar(pathLossExp) || pathLossExp<0
                disp('ERROR! input pathLossExp must be a real scalar positive');
                disp('your pathLossExp :'); disp(pathLossExp);
                error(' ');
            end
            if ~isreal(freq) || ~isscalar(freq) || freq<0
                disp('ERROR! input freq must be a real scalar positive');
                disp('your freq :'); disp(freq);
                error(' ');
            end
            Ld = pathLossRef*((dist/distRef)^(-pathLossExp));   %disp(['Ld : ', num2str(Ld)]);
            waveLen = 3e8/freq;                                 %disp(['waveLen : ', num2str(waveLen)]);
            k = 2*pi/waveLen;                                   %disp(['k : ', num2str(k)]);
            N_ar = chSize(1); rowCnt = (0:(N_ar-1)).';          %disp('N_ar :'); disp(N_ar);
            N_at = chSize(2); colCnt = (0:(N_at-1)).';          %disp('N_at :'); disp(N_at);
            d_a = waveLen/2;                                    %disp('d_a :'); disp(d_a);
            ar = (1/sqrt(N_ar))*exp(1j*k*d_a*rowCnt*sin(phi));  %disp('ar :'); disp(ar);
            at = (1/sqrt(N_at))*exp(1j*k*d_a*colCnt*sin(phi));  %disp('at :'); disp(at);
            H_NLOS = randn(chSize) + 1j*randn(chSize);          %disp('H_NLOS :'); disp(H_NLOS);
            ch = sqrt(Ld)*(sqrt(ricianFactor/(1+ricianFactor))*ar*at' + sqrt(1/(1+ricianFactor))*H_NLOS);
        end
        function lin = dB_to_linear(dB)
            lin = 10^(0.1*dB);
        end
        function watt = dBm_to_watt(dBm)
            watt = 10^(0.1*dBm - 3);
        end
    end
end