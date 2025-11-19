%% LTE 20 MHz OFDM + CFR + Oversampled Rapp PA (no DPD)
clear; clc; close all;

%% --------------------------------------------------------------------
%  0. LTE 20 MHz OFDM parameters (baseband, 30.72 MHz)
%% --------------------------------------------------------------------
Fs_bb   = 30.72e6;     % baseband sampling rate for LTE 20 MHz
Nfft    = 2048;        % IFFT size
NscUsed = 1200;        % active subcarriers (100 RB x 12)
SubSp   = 15e3;        % subcarrier spacing
BW_occu = NscUsed * SubSp;   % occupied BW ≈ 18 MHz

Ncp     = 144;         % fixed CP length
NsymOFDM= 1000;        % number of OFDM symbols
M_qam   = 16;          % 16-QAM

fprintf('Baseband: Fs = %.2f MHz, Nfft = %d, NscUsed = %d, BW_occu ≈ %.2f MHz\n',...
    Fs_bb/1e6, Nfft, NscUsed, BW_occu/1e6);

%% --------------------------------------------------------------------
%  1. 16-QAM symbols per subcarrier
%% --------------------------------------------------------------------
bitsPerSym_SC = log2(M_qam);
bitsPerOFDM   = bitsPerSym_SC * NscUsed;

bits    = randi([0 1], bitsPerOFDM * NsymOFDM, 1);
dataInt = bi2de(reshape(bits, [], bitsPerSym_SC));              % bits -> integers
qamSym  = qammod(dataInt, M_qam, 'UnitAveragePower', true);     % avg power = 1

qamGrid = reshape(qamSym, NscUsed, NsymOFDM);                   % [NscUsed x Nsym]

%% --------------------------------------------------------------------
%  2. Optional TX edge shaping (set rolloff=0 to see plain rectangular)
%% --------------------------------------------------------------------
rolloff = 0.1;                   % 10% of active band used for taper
Nedge   = floor(rolloff * NscUsed / 2);
w       = ones(NscUsed,1);

if Nedge > 0
    t    = (0:Nedge-1)'/(Nedge-1);         % 0..1
    ramp = 0.5*(1 - cos(pi*t));           % raised-cosine 0 -> 1
    w(1:Nedge)             = ramp;        % left edge
    w(end-Nedge+1:end)     = flipud(ramp);
end

qamGrid_shaped = bsxfun(@times, qamGrid, w);

%% --------------------------------------------------------------------
%  3. Map onto 2048-point IFFT grid
%% --------------------------------------------------------------------
ofdmGrid = zeros(Nfft, NsymOFDM);

% Centered mapping: -600..+599 -> 2048 bins
scIdx = (-NscUsed/2 : NscUsed/2-1);   % -600..+599
fftBin = mod(scIdx, Nfft) + 1;        % 1..2048

ofdmGrid(fftBin, :) = qamGrid_shaped;

%% --------------------------------------------------------------------
%  4. IFFT + CP  -> baseband time-domain at 30.72 MHz
%% --------------------------------------------------------------------
tx_noCP   = ifft(ofdmGrid, Nfft, 1);                 % [Nfft x Nsym]
tx_withCP = [tx_noCP(end-Ncp+1:end,:); tx_noCP];     % [(Ncp+Nfft) x Nsym]

tx_bb = tx_withCP(:);                                % serialize

%% --------------------------------------------------------------------
%  5. Normalize power and apply simple CFR (at baseband Fs_bb)
%% --------------------------------------------------------------------
tx_bb = tx_bb / sqrt(mean(abs(tx_bb).^2));    % avg power = 1

papr_lin = max(abs(tx_bb).^2) / mean(abs(tx_bb).^2);
papr_dB  = 10*log10(papr_lin);
fprintf('OFDM baseband PAPR (before CFR) ≈ %.2f dB\n', papr_dB);

% Hard clipping CFR
PAPR_target_dB = 7.5;
rms_val = sqrt(mean(abs(tx_bb).^2));
A_clip  = rms_val * 10^(PAPR_target_dB/20);

mag      = abs(tx_bb);
ang      = angle(tx_bb);
mag_clip = min(mag, A_clip);
x_cfr_bb = mag_clip .* exp(1j*ang);

papr_lin_after = max(abs(x_cfr_bb).^2) / mean(abs(x_cfr_bb).^2);
papr_dB_after  = 10*log10(papr_lin_after);
fprintf('PAPR after CFR ≈ %.2f dB\n', papr_dB_after);

% Design a simple TX low-pass filter to emulate baseband TX filtering
% Passband ~ 9 MHz, stopband ~ 12 MHz at Fs_bb = 30.72 MHz
Fp = 9e6;      % passband edge
Fsb = 12e6;    % stopband edge
Rp = 0.1;      % passband ripple (dB)
As = 60;       % stopband attenuation (dB)

% Normalized frequencies
Wp = Fp / (Fs_bb/2);
Ws = Fsb / (Fs_bb/2);

% Use a standard equiripple (or Kaiser) design
dev = [ (10^(Rp/20)-1)/(10^(Rp/20)+1)  10^(-As/20) ];
[n,fo,ao,w] = remezord([Fp Fsb],[1 0],[dev(1) dev(2)], Fs_bb);
h_tx = remez(n, fo, ao, w);   % baseband TX filter

% Filter the clipped signal (this removes most CFR spectral regrowth)
x_cfr_filt = filter(h_tx, 1, x_cfr_bb);

% Optional: renormalize power after TX filtering
x_cfr_filt = x_cfr_filt / sqrt(mean(abs(x_cfr_filt).^2));
%% --------------------------------------------------------------------
%  6. Oversample 4x before PA
%% --------------------------------------------------------------------
OSR_PA = 4;                          % oversampling ratio for PA stage
Fs_pa  = Fs_bb * OSR_PA;             % PA sampling rate => 122.88 MHz

% Polyphase resampling (anti-alias filter included)
x_cfr_pa = resample(x_cfr_filt, OSR_PA, 1);

fprintf('PA stage sampling rate Fs_pa = %.2f MHz\n', Fs_pa/1e6);

%% --------------------------------------------------------------------
%  7. Memoryless Rapp PA with AM/PM (same style as old code)
%% --------------------------------------------------------------------
A_sat  = 1.5;    % saturation amplitude
p_rapp = 2;      % smoothness factor
G_lin  = 3.0;    % small-signal gain
k_phi  = 0.3;    % AM/PM coefficient (rad)

% Drive level in dB (how hard we push into compression)
drive_dB     = 2;                       % same as your old script
drive_linear = 10^(drive_dB/20);

u = drive_linear * x_cfr_pa;            % PA input

% Rapp AM/AM and AM/PM
rappAMAM = @(r) (G_lin .* r) ./ (1 + (r./A_sat).^(2*p_rapp)).^(1/(2*p_rapp));
rappAMPM = @(r) k_phi * (r./A_sat).^2;

r_in   = abs(u);
phi_in = angle(u);
r_out  = rappAMAM(r_in);
phi_out= phi_in + rappAMPM(r_in);
y_pa   = r_out .* exp(1j*phi_out);

fprintf('[PA only] avg|u| = %.3f, peak|u| = %.3f, max/Asat = %.2f\n',...
    mean(r_in), max(r_in), max(r_in)/A_sat);

%% --------------------------------------------------------------------
%  8. Power normalization: keep PA output power comparable to input
%% --------------------------------------------------------------------
P_in  = mean(abs(u).^2);
P_out = mean(abs(y_pa).^2);
alpha = sqrt(P_in / P_out);

y_pa_norm = alpha * y_pa;   % normalized PA output

%% --------------------------------------------------------------------
%  9. Spectra: input vs PA output (shoulder now looks like old plot)
%% --------------------------------------------------------------------
Nfft_psd = 8192;
[pxx_in, f]   = pwelch(u,         hamming(4096), 2048, Nfft_psd, Fs_pa, 'centered');
[pxx_pa, ~]   = pwelch(y_pa_norm, hamming(4096), 2048, Nfft_psd, Fs_pa, 'centered');
[pxx_bb, ~]   = pwelch(x_cfr_pa,  hamming(4096), 2048, Nfft_psd, Fs_pa, 'centered');

figure;
plot(f/1e6, 10*log10(pxx_pa + eps), 'r', 'LineWidth', 1); hold on;
plot(f/1e6, 10*log10(pxx_in + eps), 'b', 'LineWidth', 1);
plot(f/1e6, 10*log10(pxx_bb + eps), 'g', 'LineWidth', 1);
grid on;
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz, normalized)');
title('Spectrum: PA input vs PA output (LTE 20 MHz, oversampled)');
legend('PA output','PA input','CFR output (upsampled)','Location','SouthWest');
xlim([-50 50]);     % now you have room to see the “shoulder” rise

%% --------------------------------------------------------------------
% 10. Effective AM/AM & AM/PM (u -> y_pa_norm)
%% --------------------------------------------------------------------
valid_idx = (1:length(u)).';     % no explicit memory here

r_in_eff  = abs(u(valid_idx));
r_out_eff = abs(y_pa_norm(valid_idx));

figure;
plot(r_in_eff, r_out_eff, '.', 'MarkerSize', 2);
grid on;
xlabel('Input amplitude |u|');
ylabel('Output amplitude |y_{PA}| (normalized)');
title('Effective AM/AM: memoryless Rapp PA');

phi_eff = angle(y_pa_norm(valid_idx) .* conj(u(valid_idx)));   % phase(y/x)

% Bin-average for smoother AM/PM
nbins   = 60;
edges   = linspace(min(r_in_eff), max(r_in_eff), nbins+1);
r_bin   = nan(nbins,1);
phi_bin = nan(nbins,1);
for k = 1:nbins
    idx = (r_in_eff >= edges(k) & r_in_eff < edges(k+1));
    if any(idx)
        r_bin(k)   = mean(r_in_eff(idx));
        phi_bin(k) = mean(phi_eff(idx));
    end
end

figure;
plot(r_in_eff, phi_eff, '.', 'MarkerSize', 1); hold on;
plot(r_bin, phi_bin, 'r-', 'LineWidth', 2);
grid on;
xlabel('Input amplitude |u|');
ylabel('Phase shift (rad)');
title('Effective AM/PM: memoryless Rapp PA');
legend('Scatter','Binned average','Location','SouthEast');

%% --------------------------------------------------------------------
% 11. Simple 3GPP-style ACLR for LTE 20 MHz
%% --------------------------------------------------------------------
BW_main = 18e6;          % ≈ occupied BW
ACLR_in  = estimateACLR_3gpp(u,         Fs_pa, BW_main);
ACLR_paN = estimateACLR_3gpp(y_pa_norm, Fs_pa, BW_main);

fprintf('\n[ACLR] PA input : L = %.2f dB, R = %.2f dB\n', ACLR_in(1),  ACLR_in(2));
fprintf('[ACLR] PA output: L = %.2f dB, R = %.2f dB\n', ACLR_paN(1), ACLR_paN(2));

%% ====================== Local functions ==============================
function ACLR = estimateACLR_3gpp(x, Fs, BW_main)
    % Very simple 3GPP-like ACLR:
    % - main channel:   f in [-BW_main/2, +BW_main/2]
    % - left adjacent:  centered at -20 MHz, same BW
    % - right adjacent: centered at +20 MHz, same BW

    Nfft = 8192;
    [pxx,f] = pwelch(x, hamming(4096), 2048, Nfft, Fs, 'centered');

    % Main band
    idx_main = (f >= -BW_main/2 & f <= BW_main/2);
    P_main   = sum(pxx(idx_main));

    % Channel spacing for LTE 20 MHz is 20 MHz
    f_off = 20e6;

    % Left adjacent
    idx_L = (f >= -f_off - BW_main/2) & (f <= -f_off + BW_main/2);
    P_L   = sum(pxx(idx_L));

    % Right adjacent
    idx_R = (f >=  f_off - BW_main/2) & (f <=  f_off + BW_main/2);
    P_R   = sum(pxx(idx_R));

    ACLR_L = 10*log10(P_main / (P_L + eps));
    ACLR_R = 10*log10(P_main / (P_R + eps));

    ACLR = [ACLR_L, ACLR_R];
end
