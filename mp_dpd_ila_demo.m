%% ========================================================================
%  Single-carrier 16QAM + Rapp PA + Iterative GMP-DPD (ILA)
%  Include：LTE baseband generation, CFR, Rapp PA(AM/AM+AM/PM), GMP-DPD, Iterative ILA
% ========================================================================
clear; clc; close all;

%% ----------------- Basic Parameters -----------------
M       = 16;      % 16QAM
Nsym    = 5000;    % baseband symbol count
sps     = 8;       % samples per symbol (oversampling factor)
beta    = 0.22;    % RRC roll-off
span    = 10;      % RRC filter span (in symbols)

Fs      = 30.72e6;     % sample rate (LTE-ish 20 MHz)
Rs      = Fs / sps;    % symbol rate

%% ----------------- Generate 16QAM baseband symbols -----------------
data = randi([0 M-1], Nsym, 1);
sym  = qammod(data, M, 'UnitAveragePower', true);   % avg power = 1

%% ----------------- TX RRC filtering & upsampling -------------------
rrcTx = rcosdesign(beta, span, sps, 'sqrt');    % Tx RRC (root raised cosine)

tx_upsampled = upsample(sym, sps);
tx_bb = conv(tx_upsampled, rrcTx, 'full');      % baseband signal (before drive)

% Remove group-delay transients
delay_rrc = span * sps / 2;
tx_bb = tx_bb(delay_rrc+1 : end-delay_rrc);

% Normalize average power to 1 （control PA drive）
tx_bb = tx_bb / sqrt(mean(abs(tx_bb).^2));

%% --------- Hard clipping CFR (limit PAPR) ---------
rms_val = sqrt(mean(abs(tx_bb).^2));

PAPR_target_dB = 7.5;                   
A_clip = rms_val * 10^(PAPR_target_dB/20);

mag  = abs(tx_bb);
ang  = angle(tx_bb);
mag_clip = min(mag, A_clip);
x_cfr = mag_clip .* exp(1j*ang);        % baseband after CFR

PAPR_before = 20*log10(max(mag)/rms_val);
PAPR_after  = 20*log10(max(mag_clip)/rms_val);
fprintf('PAPR before CFR = %.2f dB, after CFR = %.2f dB\n', ...
    PAPR_before, PAPR_after);

%% ----------------- Rapp PA model parameters ------------------------
A_sat  = 1.2;   % saturation amplitude
p_rapp = 3;     % smoothness factor
G_lin  = 4;     % small-signal gain
k_phi  = 0.4;   % AM/PM coeff

drive_dB     = 1;                      
drive_linear = 10^(drive_dB/20);

u = drive_linear * x_cfr;     

%% ----------------- PA model ---------------------------
PA = @(x) rappPA(x, A_sat, p_rapp, G_lin, k_phi);

% 无 DPD 的 PA 输出（baseline）
y_pa_nodp = PA(u);

fprintf('\n--- PA only statistics ---\n');
r_in = abs(u);
fprintf('Avg |x| = %.3f, Peak |x| = %.3f, Max/Asat = %.2f\n', ...
    mean(r_in), max(r_in), max(r_in)/A_sat);

%% ----------------- GMP DPD Parameters -------------------------------
P_main  = [1 3 5 7];   % odd orders
M_main  = 5;           % memory depth (main)

% Cross-term 分支 
P_cross = [3 5];      
M_cross = 3;           % memory depth for cross terms
L_lag   = 2;           % lag between main & cross branch

nIter   = 5;

[~, ~, offset_gmp, numCols] = buildGMP(u, P_main, M_main, P_cross, M_cross, L_lag);

c_gmp = zeros(numCols, 1);

%% ----------------- Iterative ILA loop for GMP-DPD -------------------
x_dpd = u;           
y_pa_dpd = y_pa_nodp; 

ACLR_hist = zeros(nIter+1, 2);  % [ACLR_L, ACLR_R] per iteration

ACLR_hist(1,:) = estimateACLR(y_pa_nodp, Fs);

for it = 1:nIter
    fprintf('\n===== ILA Iteration %d =====\n', it);

    y_pa_iter = PA(x_dpd);

    [Phi_y, ~, offset, ~] = buildGMP(y_pa_iter, P_main, M_main, ...
                                     P_cross, M_cross, L_lag);

    [~, u_eff, ~, ~] = buildGMP(u, P_main, M_main, ...
                                P_cross, M_cross, L_lag);
    x_target = u_eff;     

    c_gmp = (Phi_y' * Phi_y) \ (Phi_y' * x_target);

    [Phi_u, u_eff2, offset2, ~] = buildGMP(u, P_main, M_main, ...
                                           P_cross, M_cross, L_lag);
    x_dpd_eff = Phi_u * c_gmp;      

    x_dpd = u;
    x_dpd(offset2+1:end) = x_dpd_eff;

    y_pa_temp = PA(x_dpd);

    pow_pa_only = mean(abs(y_pa_nodp).^2);
    pow_pa_dpd  = mean(abs(y_pa_temp).^2);
    alpha = sqrt(pow_pa_only / pow_pa_dpd);

    x_dpd      = alpha * x_dpd;
    y_pa_dpd   = alpha * y_pa_temp;

    ACLR_hist(it+1,:) = estimateACLR(y_pa_dpd, Fs);
    fprintf('ACLR (L, R) = (%.2f dB, %.2f dB)\n', ACLR_hist(it+1,1), ACLR_hist(it+1,2));
end

%% ----------------- Effective AM/AM & AM/PM --------------------------
valid_idx = (offset_gmp+1:length(u)).';

r_in_pa_eff   = abs(u(valid_idx));
r_out_pa_eff  = abs(y_pa_nodp(valid_idx));

r_in_dpd_eff  = abs(u(valid_idx));
r_out_dpd_eff = abs(y_pa_dpd(valid_idx));

figure;
plot(r_in_pa_eff,  r_out_pa_eff,  'b.', 'MarkerSize', 4); hold on;
plot(r_in_dpd_eff, r_out_dpd_eff, 'r.', 'MarkerSize', 4);
grid on; axis tight;
xlabel('Input amplitude (|u|)');
ylabel('Output amplitude (|y|)');
legend('PA only','PA + GMP-DPD','Location','SouthEast');
title('Effective AM/AM: u -> y');

phi_pa_only_eff = angle(y_pa_nodp(valid_idx) .* conj(u(valid_idx)));
phi_pa_dpd_eff  = angle(y_pa_dpd(valid_idx)   .* conj(u(valid_idx)));

figure;
plot(r_in_pa_eff, phi_pa_only_eff, 'b.', 'MarkerSize', 4); hold on;
plot(r_in_dpd_eff, phi_pa_dpd_eff,  'r.', 'MarkerSize', 4);
grid on;
xlabel('Input amplitude (|u|)');
ylabel('Phase shift (rad)');
legend('PA only','PA + GMP-DPD','Location','SouthEast');
title('Effective AM/PM: u -> y');

%% ----------------- Constellation (with ideal RX RRC) ----------------
rrcRx = rrcTx;  % matched filter

% Ideal channel (no PA, no DPD, only CFR)
y_ideal = conv(x_cfr, rrcRx, 'full');
% PA only
y_pa_rx = conv(y_pa_nodp, rrcRx, 'full');
% PA + GMP-DPD
y_pa_dpd_rx = conv(y_pa_dpd, rrcRx, 'full');

delay2 = span * sps;    % Tx + Rx total delay
y_ideal     = y_ideal(    delay2+1 : delay2 + Nsym*sps);
y_pa_rx     = y_pa_rx(    delay2+1 : delay2 + Nsym*sps);
y_pa_dpd_rx = y_pa_dpd_rx(delay2+1 : delay2 + Nsym*sps);

% Downsample to symbol rate
y_ideal_sym   = y_ideal(1:sps:end);
y_pa_sym      = y_pa_rx(1:sps:end);
y_pa_dpd_sym  = y_pa_dpd_rx(1:sps:end);

figure;
subplot(1,3,1);
plot(real(y_ideal_sym), imag(y_ideal_sym), '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('Constellation (Ideal, CFR only)');

subplot(1,3,2);
plot(real(y_pa_sym), imag(y_pa_sym), '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('After PA (no DPD)');

subplot(1,3,3);
plot(real(y_pa_dpd_sym), imag(y_pa_dpd_sym), '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('After PA + GMP-DPD');

%% ----------------- Spectra: PA only vs PA+GMP-DPD -------------------
Nfft_psd = 4096;
[pxx_in,f]       = pwelch(u,          hamming(2048), 1024, Nfft_psd, Fs, 'centered');
[pxx_out,~]      = pwelch(y_pa_nodp,  hamming(2048), 1024, Nfft_psd, Fs, 'centered');
[pxx_out_dpd,~]  = pwelch(y_pa_dpd,   hamming(2048), 1024, Nfft_psd, Fs, 'centered');

figure;
plot(f/1e6, 10*log10(pxx_out   + eps), 'r', 'LineWidth', 1); hold on;
plot(f/1e6, 10*log10(pxx_out_dpd + eps), 'g', 'LineWidth', 1);
plot(f/1e6, 10*log10(pxx_in    + eps), 'b', 'LineWidth', 1);
grid on;
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz, normalized)');
title('Spectrum: PA only vs PA + GMP-DPD (Iterative ILA)');
legend('PA only','PA + GMP-DPD','Input to PA','Location','SouthWest');
xlim([-10 10]);

%% ----------------- Optional: ACLR convergence plot ------------------
figure;
plot(0:nIter, ACLR_hist(:,1), '-o'); hold on;
plot(0:nIter, ACLR_hist(:,2), '-s');
grid on;
xlabel('Iteration');
ylabel('ACLR (dBc)');
legend('Left','Right','Location','SouthWest');
title('ACLR convergence (GMP-DPD, ILA)');

%% ====================================================================
%  Local function: Rapp PA with AM/AM + AM/PM
% ====================================================================
function y = rappPA(x, A_sat, p_rapp, G_lin, k_phi)
    x = x(:);
    r = abs(x);
    phi = angle(x);

    % Rapp AM/AM
    r_out = (G_lin .* r) ./ (1 + (r./A_sat).^(2*p_rapp)).^(1/(2*p_rapp));

    % Simple AM/PM model
    phi_out = phi + k_phi * (r./A_sat).^2;

    y = r_out .* exp(1j*phi_out);
end

%% ====================================================================
%  Local function: buildGMP
%  x:        complex input vector
%  P_main:   main branch orders, e.g., [1 3 5 7]
%  M_main:   memory depth of main branch
%  P_cross:  cross-term orders, e.g., [3 5]
%  M_cross:  memory depth of cross branch
%  L_lag:    number of lags between main & cross
%
%  输出:
%    Phi:      [N_eff x numCols] regression matrix
%    x_eff:    x(offset+1 : end)
%    offset:   effective start address
%    numCols:  GMP column
% ====================================================================
function [Phi, x_eff, offset, numCols] = buildGMP(x, P_main, M_main, ...
                                                 P_cross, M_cross, L_lag)
    x = x(:);
    N = length(x);

    L1 = M_main;   % main memory
    L2 = M_cross;  % cross memory
    numP1 = length(P_main);
    numP2 = length(P_cross);

    offset = max(L1-1, L2-1+L_lag);
    N_eff  = N - offset;

    numCols = L1*numP1 + L2*L_lag*numP2;
    Phi   = zeros(N_eff, numCols);
    x_eff = x(offset+1:end);

    for n = offset+1:N
        row = n - offset;
        col = 1;

        % ---- main branch: x[n-m] |x[n-m]|^{p-1} ----
        for m = 0:L1-1
            xm = x(n-m);
            ax = abs(xm);
            for k = 1:numP1
                p = P_main(k);
                Phi(row, col) = xm * ax^(p-1);
                col = col + 1;
            end
        end

        % ---- cross terms branch (GMP) ----
        % x[n-m-l] * |x[n-m]|^{p-1}
        for m = 0:L2-1
            x_ref = x(n-m);          % reference tap (envelope)
            a_ref = abs(x_ref);
            for l = 1:L_lag
                x_delayed = x(n-m-l); % delayed tap
                for k = 1:numP2
                    p = P_cross(k);
                    Phi(row, col) = x_delayed * a_ref^(p-1);
                    col = col + 1;
                end
            end
        end
    end
end

%% ====================================================================
%  Local function: simple ACLR estimate
% ====================================================================
function ACLR = estimateACLR(x, Fs)
    Nfft = 8192;
    [pxx,f] = pwelch(x, hamming(2048), 1024, Nfft, Fs, 'centered');

    BW_main = 10e6;
    BW_adj  = 10e6;

    idx_main = (f >= -BW_main/2 & f <= BW_main/2);
    P_main   = sum(pxx(idx_main));

    idx_L = (f >= -3*BW_main & f <= -BW_main);
    P_L   = sum(pxx(idx_L));

    idx_R = (f >=  BW_main & f <=  3*BW_main);
    P_R   = sum(pxx(idx_R));

    ACLR_L = 10*log10(P_main / P_L + eps);
    ACLR_R = 10*log10(P_main / P_R + eps);

    ACLR = [ACLR_L, ACLR_R];
end

