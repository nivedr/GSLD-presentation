% GSLD
clear;
clc;

n_t    = 32;   % received vector dimension (y_c)
n_r    = 32;   % transmitted vector dimension (x_c)

I_max  = 1000;
T1     = 500;
T2     = 500;

theta  = 0.1;
ALPHABET = [-3,-1,1,3];

%% Model : y_c = H_c.x_c + n_c

% channel matrix (n_r x n_t)
H_c = randn(n_r, n_t) + 1i*randn(n_r, n_t);
% H_c = circshift(H_c,);    
H = [ real(H_c) , - imag(H_c) ; imag(H_c) , real(H_c) ];
sigma  = 0;    % sigma

SIGMAITER = 5;

SER = cell(1,SIGMAITER);
BER = cell(1,SIGMAITER);
ERR = cell(1,SIGMAITER);
scale = 1;

for sigma_iter = (1:SIGMAITER)
    err = [];
    ser = [];
    ber = [];

    % x = ALPHABET(floor( size(ALPHABET,2)*rand(2*n_t,1) ) + 1)';
    % x   = [ real(x_c) ; imag(x_c) ];
    % ALPHABET = scale*ALPHABET;

    x_t = datasample(ALPHABET, 2*n_t)';
    x_true = x_t(1:end/2) + 1i*x_t(end/2+1:end);
    %x_t = randi([0 15], 1, 2*n_t);
        
    % x_t = (ALPHABET(ceil(size(ALPHABET,2)*rand(2*n_r,1))))';
    N = sigma*randn(2*n_r,1);
    y = H*x_t + N;
    %y   = [ real(y_c) ; imag(y_c) ];

    %%
    for SER_iter = (1:10)
        x = datasample(ALPHABET, 2*n_t)';

        C = 0;
        S = 0;
        t = 1;

        beta = norm(y - H*x,2);
        z = x;

        for i = (1:2*n_t)
            r(i) = norm(H(:,i),2).^2;
        end

        y_h = y - H*x;
        while t < I_max+1
            for i = (1:2*n_t)
                if C < 2*n_t
                    y_t(:,i) = y_h + H(:,i)*x(i);
                    mean_est(i) = y_t(:,i)'*H(:,i)/r(i);
                end

                s(i) = sigma/sqrt(2*r(i))*randn + mean_est(i);
                [~, x_new(i)] = min(abs(s(i)*ones(1,length(ALPHABET)) - ALPHABET));
                x_new(i) = ALPHABET(x_new(i));

                if x_new(i) ~= x(i)
                    C = 0;
                    y_h = y_t(:,i) - H(:,i)*x_new(i);
                else
                    C = C + 1;
                end
                x(i) = x_new(i);
            end

            gamma(t) = norm(y - H*x, 2);
            if gamma(t) < beta
                z = x;
                beta = gamma(t);
                S = 0;
            else
                S = S + 1;
            end
            if beta < scale*theta
                if S >= T1
                    break;
                end
            else
                if S >= T2
                    break;
                end
            end
            t = t + 1;
        end
        ser = [ ser ; sum(z-x_t==0)/(2*n_t) ];
        
        z_true = z(1:end/2) + 1i*z(end/2+1:end);
        err = [ err ; norm(y-H*x,2) ];
        ber = [ ber ; compute_BER(z_true,x_true,scale)/(length(ALPHABET)*n_t) ];
    end
    [~, position] = min(err);
    ERR{sigma_iter} = [ ERR{sigma_iter} ; min(err) ];
    SER{sigma_iter} = [ SER{sigma_iter} ; ser(position) ];
    BER{sigma_iter} = [ BER{sigma_iter} ; ber(position) ];

end

ttt = 1

