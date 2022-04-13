classdef DsmMimo < handle
    methods(Static)
        %------- DSM -------%
        function R = main(G, H, F, P, sig2, init_theta, iterLimit)
            DsmMimo.filter(G, H, F, P, sig2, init_theta);
            
            tol = 1e-3;
            [~, thetaOpt, ~, ~] = DsmMimo.maximize_psi_dsm(G, H, F, init_theta, iterLimit, tol);
            
            Heff = DsmMimo.Heff(F, G, H, thetaOpt);
            [~, Lambda, ~] = svd(Heff); lambda = diag(Lambda);
            U = rank(Heff);
            heightIn = sig2*U./(P * lambda.^2);
            [pow_alloc, ~] = DsmMimo.waterFilling(ones(size(heightIn)), heightIn, U);
            R = DsmMimo.getRate(P, pow_alloc, lambda, sig2, U);
        end
        function [psi, thetaOpt, nIter, psiTrace, timeExec] = maximize_psi_dsm(G, H, F, init_theta, iterLimit, tol)
            DsmMimo.filter(G, H, F, 1, 1, init_theta);
            if ~isequal(size(init_theta), [size(H,1) 1])
                disp('ERROR!! the size of init_theta is incorrect');
                error(' ');
            end
            tic;
            N = size(H, 1);
            thetaOpt = init_theta;
            dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, thetaOpt); err=norm(dpsi_dtheta);
            psiTrace = zeros(iterLimit+1, 1);
            psiTrace(1,1) = DsmMimo.get_psi(F, G, H, thetaOpt);
            iter_th = 0;
            GFH = G'*F*H'; GG = G'*G; HH = H*H';
            while (err>tol && iter_th<iterLimit)
                iter_th = iter_th + 1;
                for n=1:N
                    thetaOpt(n) = DsmMimo.thetaMax(GG(:,n), HH(:,n), GFH(n,n), thetaOpt, n);
                end
                psiTrace(iter_th+1, 1) = DsmMimo.get_psi(F, G, H, thetaOpt);
                dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, thetaOpt);
                err = norm(dpsi_dtheta);
            end
            psiTrace(iter_th+2:end) = [];
            thetaOpt = DsmMimo.normalizeAngles(thetaOpt);
            psi = DsmMimo.get_psi(F, G, H, thetaOpt);
            nIter = iter_th;
            timeExec = toc;
        end
        
        %------- GA -------%
        function R = main_GA1(G, H, F, P, sig2, init_theta, alpha, iterLimit)
            DsmMimo.filter(G, H, F, P, sig2, init_theta);
            if ~isscalar(alpha) || ~isreal(alpha) || alpha<0
                disp('ERROR! alpha must be a scalar real positive');
                disp('Your alpha ='); disp(alpha);
                error(' ');
            end
            tol = 1e-3;
            [~, thetaOpt, ~, ~] = DsmMimo.maximize_psi_GA1(G, H, F, init_theta, alpha, iterLimit, tol);
            
            Heff = DsmMimo.Heff(F, G, H, thetaOpt);
            [~, Lambda, ~] = svd(Heff); lambda = diag(Lambda);
            U = rank(Heff);
            heightIn = sig2*U./(P * lambda.^2);
            [pow_alloc, ~] = DsmMimo.waterFilling(ones(size(heightIn)), heightIn, U);
            R = DsmMimo.getRate(P, pow_alloc, lambda, sig2, U);
        end
        function [psi, thetaOpt, nIter, psiTrace, timeExec] = maximize_psi_GA1(G, H, F, init_theta, alpha, iterLimit, tol)
            DsmMimo.filter(G, H, F, 1, 1, init_theta);
            if ~isequal(size(init_theta), [size(H,1) 1])
                disp('ERROR!! the size of init_theta is incorrect');
                error(' ');
            end
            if ~isscalar(alpha) || ~isreal(alpha) || alpha<=0
                disp('ERROR! input alpha must be scalar, real, and positive');
                disp('Your input alpha:'); disp(alpha);
                error(' ');
            end
            tic;
            N = size(H, 1);
            thetaOpt = init_theta;
            psiTrace = zeros(iterLimit+1, 1);
            psiTrace(1,1) = DsmMimo.get_psi(F, G, H, thetaOpt); %disp(['psiTrace = ', num2str(psiTrace(1, 1))]);
            dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, thetaOpt); err=norm(dpsi_dtheta); %disp(['err = ', num2str(err)]);
            iter_th = 0;
            while (err>tol && iter_th<iterLimit)
                iter_th = iter_th + 1; %disp(['iter_th : ', num2str(iter_th), '-------']);
                for n=1:N
                    dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, thetaOpt);
                    thetaOpt(n) = thetaOpt(n) + alpha*dpsi_dtheta(n);
                    thetaOpt(n) = DsmMimo.normalizeAngles(thetaOpt(n));
                end
                psiTrace(iter_th+1, 1) = DsmMimo.get_psi(F, G, H, thetaOpt); %disp(['   psiTrace = ', num2str(psiTrace(iter_th+1, 1))]);
                dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, thetaOpt);
                err = norm(dpsi_dtheta); %disp(['   err : ', num2str(err)]);
            end
            psiTrace(iter_th+2:end)=[];
            thetaOpt = DsmMimo.normalizeAngles(thetaOpt);
            psi = DsmMimo.get_psi(F, G, H, thetaOpt);
            nIter = iter_th;
            timeExec = toc;
        end
     
        %------- Verifications -------%
        function checkEverything(G, H, F, P, sig2, init_theta)
            N = size(H,1); disp(['N = ', num2str(N)]);
            h = 1e-5;
            
            disp('----------- Test psi/dTheta -----------');
            psi_a = DsmMimo.get_psi(F, G, H, init_theta);
            dpsi_dtheta = zeros(size(init_theta));
            for n=1:N
                thetaTemp = init_theta; thetaTemp(n) = thetaTemp(n) + h;
                psi_b = DsmMimo.get_psi(F, G, H, thetaTemp);
                dpsi_dtheta(n) = (psi_b - psi_a)/h;
            end
            disp('dpsi_dtheta using limit definition :'); disp(dpsi_dtheta);
            dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, init_theta);
            disp('dpsi_dtheta using equation :'); disp(dpsi_dtheta);
            
            disp('## Check optimal theta equation');
            GFH = G'*F*H'; GG = G'*G; HH = H*H';
            thetaMax = init_theta; disp('theta_init :'); disp(init_theta);
            repeat_num = 3;
            for repeat_th = 1:repeat_num
                for n=1:N
                    disp(['n = ', num2str(n), ' ---------------------']);
                    thetaMax(n) = DsmMimo.thetaMax(GG(:,n), HH(:,n), GFH(n,n), thetaMax, n);
                    dpsi_dtheta = DsmMimo.dpsi_dtheta(F, G, H, thetaMax);
                    disp('   theta_current :'); disp(thetaMax);
                    disp('   dpsi/dtheta_n : '); disp(dpsi_dtheta);
                    disp(['   norm dpsi/dtheta :', num2str(norm(dpsi_dtheta))]);
                    psi = DsmMimo.get_psi(F, G, H, thetaMax); disp(['   psi = ', num2str(psi)]);
                end
            end
        end
        % The following two functions are protections from the wrong input
        % format, similar to "assert" in the python programming language.
        function [K, L, N] = filter(G, H, F, P, sig2, init_theta)
            if ~isscalar(P) || ~isreal(P) || P<0
                disp('ERROR! P must be a positive real scalar');
                disp('Your input P :'); disp(P);
                error(' ');
            end
            if ~isscalar(sig2) || ~isreal(sig2) || sig2<0
                disp('ERROR! sig2 must be a positive real scalar');
                disp('Your input sig2 :'); disp(sig2);
                error(' ');
            end
            if ~isequal(size(F,1), size(G,1)) %L
                disp('ERROR! row size of F and row size of G must be the same');
                disp('Your C size :'); disp(size(F));
                disp('Your A size :'); disp(size(G));
                error(' ');
            end
            if ~isequal(size(F,2), size(H,2)) %K
                disp('ERROR! column size of F and column size of H must be the same');
                disp('Your C size :'); disp(size(F));
                disp('Your B size :'); disp(size(H));
                error(' ');
            end
            if ~isequal(size(H,1), size(G,2))%N
                disp('ERROR! row size of H and column size of G must be the same');
                disp('Your H size :'); disp(size(H));
                disp('Your G size :'); disp(size(G));
                error(' ');
            end
            if ~isequal(size(init_theta), [size(H,1), 1])
                disp('ERROR! the size of init_theta must be :');
                disp([size(H,1), 1]);
                disp('your init_theta :'); disp(init_theta);
                error(' ');
            end
            if any(init_theta<-pi) || any(init_theta>pi)
                disp('ERROR! all entries of init_theta must be in range -pi to pi');
                disp('your init_theta ='); disp(init_theta);
                error(' ');
            end
            K = size(F, 2); L = size(F, 1); N = size(H, 1);
        end
        function filter2(stepsize, iterlim, tol, stepreduct)
            if ~isscalar(stepsize) || ~isreal(stepsize) || stepsize<=0
                disp('------- CUSTOM ERROR -------');
                disp('input stepsize must be real scalar positive');
                disp('your stepsize ='); disp(stepsize);
                error(' ');
            end
            if ~isscalar(iterlim) || ~isreal(iterlim) || iterlim<=0
                disp('------- CUSTOM ERROR -------');
                disp('input iterlim must be real scalar positive');
                disp('your iterlim ='); disp(iterlim);
                error(' ');
            end
            if ~isscalar(tol) || ~isreal(tol) || tol<=0
                disp('------- CUSTOM ERROR -------');
                disp('input tol must be real scalar positive');
                disp('your tol ='); disp(tol);
                error(' ');
            end
            if ~isscalar(stepreduct) || ~isreal(stepreduct) || stepreduct<=0 || stepreduct>=1
                disp('------- CUSTOM ERROR -------');
                disp('input stepreduct must be real scalar and has range between 0 and 1');
                disp('your stepreduct ='); disp(stepreduct);
                error(' ');
            end
        end
        
        %------- Equations -------%
        function out = get_psi(F, G, H, theta) % objective of problem (5) or (6)
            if ~isequal(size(theta), [size(H,1), 1])
                disp('ERROR! the size of init_theta must be :');
                disp([size(H,1), 1]);
                disp('your init_theta :'); disp(init_theta);
                error(' ');
            end
            Phi = diag(exp(1j*theta));
            out = (norm(F+G*Phi*H, 'fro'))^2;
        end
        function out = dpsi_dtheta(F, G, H, theta) % equation (9)
            phi = exp(1j*theta);
            Phi = diag(phi);
            mat = H*(F+G*Phi*H)'*G;
            out = -2*imag(diag(diag(mat.'))*phi);
        end
        function out = thetaMax(g_n, h_n, f_nn, theta, n) % equation (8)
            if ~iscolumn(g_n) || ~iscolumn(h_n) || ~iscolumn(theta)
                disp('ERROR! g_n, h_n, and theta must be column');
                error(' ');
            end
            if ~isscalar(f_nn) || ~isscalar(n)
                disp('f_nn and n must be a scalar');
                error(' ');
            end
            out = angle(f_nn + g_n'*diag(exp(1j*theta))*h_n - conj(g_n(n))*exp(1j*theta(n))*h_n(n));
        end
        function out = normalizeAngles(in) % bring phase into range -pi ~ pi
            out = -pi + mod(in+pi, 2*pi);
        end
        function out = Heff(F, G, H, theta) % definition of H_eff
            out = F + G*diag(exp(1j*theta))*H;
        end
        function out = getRate(P, power_alloc, lambda, sig2, U) % objective in (3)
            if ~isreal(P) || ~isscalar(P) || P<0
                disp('ERROR! input P must be a real positive');
                error(' ');
            end
            if ~isreal(sig2) || ~isscalar(sig2) || sig2<0
                disp('ERROR! input P must be a real positive');
                error(' ');
            end
            if ~isreal(U) || ~isscalar(U) || U<0
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
            out = sum(log2(1+P.*power_alloc.*(lambda.^2)./(sig2*U)));
        end
        function R = get_rate_v2(F, G, H, theta, sig2, P) % Line 9-10 of Alg. 1
            if ~isequal(size(theta), [size(H, 1), 1])
                disp('ERROR! the size of init_theta must be :');
                disp([size(H, 1), 1]);
                disp('your init_theta :'); disp(theta);
                error(' ');
            end
            if any(theta < -pi - 1e-3) || any(theta>pi+1e-3)
                disp('ERROR! all entries of theta must be in range -pi to pi');
                disp('your theta ='); disp(theta);
                error(' ');
            end
            Heff = DsmMimo.Heff(F, G, H, theta);
            [~, Lambda, ~] = svd(Heff); lambda = diag(Lambda);
            U = rank(Heff);
            heightIn = sig2*U./(P*lambda.^2);
            [pow_alloc, ~] = DsmMimo.waterFilling(ones(size(heightIn)), heightIn, U);
            R = DsmMimo.getRate(P, pow_alloc, lambda, sig2, U);
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
        
    end
end