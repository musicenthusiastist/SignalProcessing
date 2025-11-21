%% ========================================================================
%  OFDM  +  CFR  +  memoryless Rapp PA  +  MP / GMP-like DPD
%  - Baseband Fs = Nfft * SubSp, oversample x4 for PA / DPD
%  - PA: memoryless Rapp + AM/PM (your parameters)
%  - DPD: main-branch GMP (i.e. Memory Polynomial)
%  - Plots: spectra, AM/AM, AM/PM, constellation
% ========================================================================
clear; clc; close all;

%% ------------------------------------------------------------------------
% 0. System configuration: bandwidth etc.
% -------------------------------------------------------------------------
BW_MHz       = 100;      % 20 / 40 / 100 ...
NsymOFDM     = 400;     % OFDM symbols
M_qam        = 16;      % 16-QAM
SubSp        = 15e3;    % SCS 15 kHz
Ncp          = 144;     % Length of CP

% Choose Nfft
switch BW_MHz
    case 20
        Nfft = 2048;
    case 40
        Nfft = 4096;
    case 100
        Nfft = 8192;
    otherwise
        error('BW_MHz = %d undefined，add Nfft manually to switch.', BW_MHz);
end

Fs_base = Nfft * SubSp;           % Sample rate
RB_total = floor((BW_MHz*1e6) / (12*SubSp));   
NscUsed  = RB_total * 12;                      
BW_occu  = NscUsed * SubSp;                    

fprintf('OFDM: BW=%.1f MHz, Fs=%.2f MHz, Nfft=%d, NscUsed=%d (RB=%d), BW_occu≈%.2f MHz, Nsym=%d\n', ...
    BW_MHz, Fs_base/1e6, Nfft, NscUsed, RB_total, BW_occu/1e6, NsymOFDM);

%% ------------------------------------------------------------------------
% 1. Generate OFDM baseband 
% -------------------------------------------------------------------------
bitsPerSC    = log2(M_qam);                % bits / subcarrier
bitsPerOFDM  = bitsPerSC * NscUsed;        % bits / OFDM symbol

bits   = randi([0 1], bitsPerOFDM * NsymOFDM, 1);
dataInt = bi2de(reshape(bits, [], bitsPerSC));
qamSym  = qammod(dataInt, M_qam, 'UnitAveragePower', true);  % avg power=1

qamGrid = reshape(qamSym, NscUsed, NsymOFDM);  % [NscUsed x NsymOFDM]

% taper（raised-cosine TX filter）
rolloff = 0.2;                           % 20% taper
Nedge   = floor(rolloff * NscUsed / 2);  % bins per edge
w = ones(NscUsed,1);
if Nedge > 0
    t    = (0:Nedge-1)'/(Nedge-1);
    ramp = 0.5*(1 - cos(pi*t));          % 0→1 raised cosine
    w(1:Nedge)               = ramp;
    w(end-Nedge+1:end)       = flipud(ramp);
end
qamGrid = qamGrid .* w;

% Map onto Nfft IFFT grid (centered NscUsed subcarriers)
ofdmGrid = zeros(Nfft, NsymOFDM);
scIdx = (-NscUsed/2 : NscUsed/2-1);     % symmetric around DC
fftBin = mod(scIdx, Nfft) + 1;
ofdmGrid(fftBin, :) = qamGrid;

% IFFT (time domain)
tx_noCP = ifft(ofdmGrid, Nfft, 1);      % [Nfft x Nsym]

% Add CP
tx_withCP = [tx_noCP(end-Ncp+1:end,:); tx_noCP];  % [(Ncp+Nfft) x Nsym]
tx_bb = tx_withCP(:);                               % serialize

% Normalize power
tx_bb = tx_bb / sqrt(mean(abs(tx_bb).^2));

% PAPR before CFR
papr_lin = max(abs(tx_bb).^2) / mean(abs(tx_bb).^2);
papr_dB  = 10*log10(papr_lin);
fprintf('Baseband PAPR before CFR ≈ %.2f dB\n', papr_dB);

%% ------------------------------------------------------------------------
% 2. Simple CFR: hard clipping to target PAPR
% -------------------------------------------------------------------------
PAPR_target_dB = 7.5;
rms_val = sqrt(mean(abs(tx_bb).^2));
A_clip  = rms_val * 10^(PAPR_target_dB/20);

mag     = abs(tx_bb);
ang     = angle(tx_bb);
mag_clip = min(mag, A_clip);
x_cfr    = mag_clip .* exp(1j*ang);

papr_lin_after = max(abs(x_cfr).^2) / mean(abs(x_cfr).^2);
papr_dB_after  = 10*log10(papr_lin_after);
fprintf('PAPR after CFR ≈ %.2f dB\n', papr_dB_after);

%% ------------------------------------------------------------------------
% 3. Oversample for PA / DPD (OSR = 4)
% -------------------------------------------------------------------------
OSR      = 4;
Fs_overs = Fs_base * OSR;

% Use resample() (with anti-aliasing filter) for upsampling
x_cfr_os = resample(x_cfr, OSR, 1);     % CFR output at Fs_overs

A_sat  = 2.5;      % saturation amplitude
p_rapp = 2;        % smoothness factor
G_lin  = 3;        % small-signal linear gain
k_phi  = 0.3;      % AM/PM coefficient (rad)

% Drive level (how far into compression)
drive_dB     = 3;
drive_linear = 10^(drive_dB/20);

% PA input (after CFR + oversampling + drive)
u = drive_linear * x_cfr_os;

% Rapp AM/AM and AM/PM (static nonlinearity)
rappAMAM = @(r) (G_lin .* r) ./ (1 + (r./A_sat).^(2*p_rapp)).^(1/(2*p_rapp));
rappAMPM = @(r) k_phi * (r./A_sat).^2;

% Simple memory filter (FIR) to emulate PA memory
h_mem = [1 0.4 0.2].';   
memPa = length(h_mem) - 1;

% PA model: static Rapp nonlinearity followed by linear memory filter
PA = @(x) rappPA_with_memory(x, rappAMAM, rappAMPM, h_mem);

% PA output (baseline, no DPD)
y_pa_nodp = PA(u);

fprintf('[PA only] avg|u|=%.3f, peak|u|=%.3f, max/Asat=%.2f\n', ...
    mean(abs(u)), max(abs(u)), max(abs(u))/A_sat);

%% ------------------------------------------------------------------------
% 5. GMP DPD (ILA)
% -------------------------------------------------------------------------
% x_DPD[n] = sum_{m,k} c_{m,k} u[n-m] |u[n-m]|^{p_k-1}
% + cross terms that use delayed envelope (true GMP)

P_main  = [1 3 5 7];   % odd orders for main branch
M_main  = 5;           % memory depth (taps) for main branch

% Cross-term (GMP) parameters 
P_cross = [3 5];       % cross-term orders (envelope-based)
M_cross = 3;           % cross-term memory depth
L_lag   = 1;           % lag between signal and envelope

nIter   = 2;           % ILA iterations
decim   = 8;           % decimation for LS (every 8th sample)

idx_train = 1:decim:length(u);  % training indices

x_dpd    = u;                  % initial DPD = identity
y_pa_dpd = y_pa_nodp;

ACLR_hist = zeros(nIter+1, 2);
ACLR_hist(1,:) = estimateACLR_simple(y_pa_nodp, Fs_overs, BW_occu);

fprintf('\n[Iteration 0] ACLR (L,R) = (%.2f, %.2f) dB\n', ...
    ACLR_hist(1,1), ACLR_hist(1,2));

for it = 1:nIter
    fprintf('\n===== GMP-DPD ILA Iteration %d =====\n', it);

    % 1) Pass current x_dpd through PA
    y_pa_iter = PA(x_dpd);

    % 2) Build GMP matrices using decimated sequences
    y_train = y_pa_iter(idx_train);
    u_train = u(idx_train);

    [Phi_y, ~, offset_mp, numCols] = buildGMP_mainBranch( ...
        y_train, P_main, M_main, P_cross, M_cross, L_lag);

    [~, u_eff, offset_u, ~] = buildGMP_mainBranch( ...
        u_train, P_main, M_main, P_cross, M_cross, L_lag);

    if offset_mp ~= offset_u
        warning('Offsets differ (%.0f vs %.0f) – check GMP builder.', ...
            offset_mp, offset_u);
    end

    x_target = u_eff;   % ideal mapping: y -> u

    % 3) Complex LS: c_mp = argmin ||Phi_y*c_mp - x_target||^2
    c_mp = (Phi_y' * Phi_y) \ (Phi_y' * x_target);

    fprintf('Number of GMP coefficients = %d\n', numCols);

    % 4) Build DPD output for the full u (no decimation)
    [Phi_u_full, u_eff_full, offset_full, ~] = buildGMP_mainBranch( ...
        u, P_main, M_main, P_cross, M_cross, L_lag);

    x_dpd_eff = Phi_u_full * c_mp;

    x_dpd = u;                         % start from original
    x_dpd(offset_full+1:end) = x_dpd_eff;

    % 5) Pass through PA again
    y_pa_temp = PA(x_dpd);

    % 6) Power alignment (keep PA+DPD power ~ PA-only power)
    pow_pa_only = mean(abs(y_pa_nodp).^2);
    pow_dpd_raw = mean(abs(y_pa_temp).^2);
    alpha = sqrt(pow_pa_only / pow_dpd_raw);

    x_dpd    = alpha * x_dpd;
    y_pa_dpd = alpha * y_pa_temp;

    % 7) ACLR
    ACLR_hist(it+1,:) = estimateACLR_simple(y_pa_dpd, Fs_overs, BW_occu);
    fprintf('ACLR (L,R) = (%.2f, %.2f) dB\n', ...
        ACLR_hist(it+1,1), ACLR_hist(it+1,2));
end

%% ------------------------------------------------------------------------
% 6. Spectra: input vs PA only vs PA+GMP-DPD
% -------------------------------------------------------------------------
Nfft_psd = 8192;
[pxx_in,f]  = pwelch(u,         hamming(4096), 2048, Nfft_psd, Fs_overs, 'centered');
[pxx_pa,~]  = pwelch(y_pa_nodp, hamming(4096), 2048, Nfft_psd, Fs_overs, 'centered');
[pxx_dpd,~] = pwelch(y_pa_dpd,  hamming(4096), 2048, Nfft_psd, Fs_overs, 'centered');

figure;
plot(f/1e6, 10*log10(pxx_pa  + eps), 'r', 'LineWidth', 1.2); hold on;
plot(f/1e6, 10*log10(pxx_dpd + eps), 'g', 'LineWidth', 1.2);
plot(f/1e6, 10*log10(pxx_in  + eps), 'b', 'LineWidth', 1.2);
grid on;
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz, normalized)');
title(sprintf('Spectrum: PA input vs PA output vs PA+GMP-DPD (%s, OSR=4)', BW_MHz));
legend('PA only','PA+GMP-DPD','PA input','Location','SouthWest');
xlim([-Fs_base Fs_base]/1e6);  % full baseband span

%% ------------------------------------------------------------------------
% 7. Effective AM/AM and AM/PM: u -> y
% -------------------------------------------------------------------------
% Because PA has memory, discard the first memPa samples
valid_idx = (memPa+1:length(u)).';

r_in   = abs(u(valid_idx));
r_out0 = abs(y_pa_nodp(valid_idx));
r_out1 = abs(y_pa_dpd(valid_idx));

% AM/AM scatter + binned average
[rb0, yf0] = binScatter(r_in, r_out0, 80);
[rb1, yf1] = binScatter(r_in, r_out1, 80);

figure;
plot(r_in, r_out0,  'b.', 'MarkerSize', 1); hold on;
plot(r_in, r_out1,  'r.', 'MarkerSize', 1);
plot(rb0, yf0, 'b', 'LineWidth', 2);
plot(rb1, yf1, 'r', 'LineWidth', 2);
grid on;
xlabel('Input amplitude |u|');
ylabel('Output amplitude |y| (normalized)');
legend('PA only scatter','PA+GMP-DPD scatter', ...
       'PA only (binned)','PA+GMP-DPD (binned)', ...
       'Location','SouthEast');
title('Effective AM/AM: u \rightarrow y');

% AM/PM
phi0 = angle(y_pa_nodp(valid_idx) .* conj(u(valid_idx)));
phi1 = angle(y_pa_dpd(valid_idx)   .* conj(u(valid_idx)));

[rbp0, phb0] = binScatter(r_in, phi0, 80);
[rbp1, phb1] = binScatter(r_in, phi1, 80);

figure;
plot(r_in, phi0, 'b.', 'MarkerSize', 1); hold on;
plot(r_in, phi1, 'r.', 'MarkerSize', 1);
plot(rbp0, phb0, 'b', 'LineWidth', 2);
plot(rbp1, phb1, 'r', 'LineWidth', 2);
grid on;
xlabel('Input amplitude |u|');
ylabel('Phase shift (rad)');
legend('PA only scatter','PA+GMP-DPD scatter', ...
       'PA only (binned)','PA+GMP-DPD (binned)', ...
       'Location','SouthEast');
title('Effective AM/PM: u \rightarrow y');

%% 8. Constellation (baseband vs PA only vs PA+GMP-DPD)

% 8.1  Baseband (CFR output) at Fs_base
x_bb_for_const = x_cfr;

qamTxGrid = ofdm_demod(x_bb_for_const, Nfft, Ncp, fftBin);

% 8.2  PA only: downsample from Fs_overs to Fs_base
y_pa_ds = y_pa_nodp(1:OSR:size(y_pa_nodp,1));
qamPaGrid = ofdm_demod(y_pa_ds, Nfft, Ncp, fftBin);

% 8.3  PA + GMP-DPD: downsample
y_pa_dpd_ds = y_pa_dpd(1:OSR:size(y_pa_dpd,1));
qamDpdGrid  = ofdm_demod(y_pa_dpd_ds, Nfft, Ncp, fftBin);

% 8.4  Pick one central subcarrier to display constellation
scMid = floor(NscUsed/2) + 1;    % roughly center of occupied band

tx_sc  = qamTxGrid(scMid, :);
pa_sc  = qamPaGrid(scMid, :);
dpd_sc = qamDpdGrid(scMid, :);

figure;
subplot(1,3,1);
plot(real(tx_sc),  imag(tx_sc),  '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('CFR output (baseband)');

subplot(1,3,2);
plot(real(pa_sc),  imag(pa_sc),  '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('After PA (no DPD)');

subplot(1,3,3);
plot(real(dpd_sc), imag(dpd_sc), '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('After PA + GMP-DPD');

%% ========================================================================
%                      LOCAL FUNCTIONS
% ========================================================================

function y = rappPA_memoryless(x, AMAM, AMPM)
    % Static Rapp PA without memory
    r   = abs(x);
    phi = angle(x);
    r_o   = AMAM(r);
    phi_o = phi + AMPM(r);
    y = r_o .* exp(1j*phi_o);
end

function y = rappPA_with_memory(x, AMAM, AMPM, h_mem)
    % Rapp PA with memory:
    %   1) static Rapp nonlinearity
    %   2) linear memory filter h_mem
    x   = x(:);
    y_nl = rappPA_memoryless(x, AMAM, AMPM);
    y    = filter(h_mem, 1, y_nl);
end

% ------------------------------------------------------------------------
% GMP-like builder (main-branch + optional cross terms)
% ------------------------------------------------------------------------
function [Phi, x_eff, offset, numCols] = buildGMP_mainBranch( ...
    x, P_main, M_main, P_cross, M_cross, L_lag)

    x = x(:);
    N = length(x);

    hasCross = ~isempty(P_cross) && M_cross > 0 && L_lag > 0;

    memMain  = M_main;
    memCross = 0;
    if hasCross
        memCross = M_cross + L_lag;
    end

    Lmax   = max(memMain, memCross);
    offset = Lmax - 1;      % number of discarded initial samples
    N_eff  = N - offset;
    if N_eff <= 0
        error('Sequence too short for given memory settings.');
    end

    numMainCols  = M_main  * length(P_main);
    numCrossCols = 0;
    if hasCross
        numCrossCols = M_cross * length(P_cross);
    end
    numCols = numMainCols + numCrossCols;

    Phi   = zeros(N_eff, numCols);
    x_eff = x((offset+1):end);

    row = 1;
    for n = (offset+1):N
        col = 1;
        % --- main branch: x[n-m] |x[n-m]|^{p-1}
        for m = 0:M_main-1
            xm  = x(n-m);
            amp = abs(xm);
            for k = 1:length(P_main)
                p = P_main(k);
                Phi(row,col) = xm * amp^(p-1);
                col = col + 1;
            end
        end

        % --- cross terms: x[n-m] |x[n-m-L_lag]|^{p-1}
        if hasCross
            for m = 0:M_cross-1
                xm      = x(n-m);
                idx_env = n - m - L_lag;
                if idx_env >= 1
                    env = abs(x(idx_env));
                else
                    env = 0;
                end
                for k = 1:length(P_cross)
                    p = P_cross(k);
                    Phi(row,col) = xm * env^(p-1);
                    col = col + 1;
                end
            end
        end
        row = row + 1;
    end
end

% ------------------------------------------------------------------------
% ACLR estimator: simple 3-band integration around 0 Hz
% BW_main: occupied main-band (≈ NscUsed * SubSp)
% ------------------------------------------------------------------------
function ACLR = estimateACLR_simple(x, Fs, BW_main)
    Nfft = 8192;
    [pxx,f] = pwelch(x, hamming(4096), 2048, Nfft, Fs, 'centered');

    BW_adj = BW_main;  % equal adjacent bandwidth

    idx_main = (f >= -BW_main/2 & f <= BW_main/2);
    P_main   = sum(pxx(idx_main));

    idx_L = (f >= -BW_main - BW_adj/2 & f <= -BW_main/2);
    idx_R = (f >=  BW_main/2 & f <=  BW_main + BW_adj/2);

    P_L = sum(pxx(idx_L));
    P_R = sum(pxx(idx_R));

    ACLR_L = 10*log10(P_main / (P_L + eps));
    ACLR_R = 10*log10(P_main / (P_R + eps));

    ACLR = [ACLR_L, ACLR_R];
end

% ------------------------------------------------------------------------
% Utility: bin scatter for smoother AM/AM & AM/PM curves
% ------------------------------------------------------------------------
function [x_bin, y_bin] = binScatter(x, y, nbins)
    x   = x(:);
    y   = y(:);
    if nargin < 3
        nbins = 50;
    end
    edges = linspace(min(x), max(x), nbins+1);
    x_bin = nan(nbins,1);
    y_bin = nan(nbins,1);
    for k = 1:nbins
        idx = (x >= edges(k) & x < edges(k+1));
        if any(idx)
            x_bin(k) = mean(x(idx));
            y_bin(k) = mean(y(idx));
        end
    end
end

% ------------------------------------------------------------------------
% OFDM demod helper for constellation plotting
% ------------------------------------------------------------------------
function qamGrid_rx = ofdm_demod(x_in, Nfft, Ncp, fftBin)
    symLen  = Nfft + Ncp;                   % samples per OFDM symbol
    Nsym_rx = floor(length(x_in)/symLen);   % number of full symbols

    x_in  = x_in(1 : Nsym_rx*symLen);
    x_mat = reshape(x_in, symLen, Nsym_rx); % [time x symbol]

    % Remove CP
    x_noCP = x_mat(Ncp+1:end, :);           % [Nfft x Nsym]

    % FFT to frequency domain
    X_fd = fft(x_noCP, Nfft, 1);            % [Nfft x Nsym]

    % Extract active subcarriers
    qamGrid_rx = X_fd(fftBin, :);           % [NscUsed x Nsym]
end
