%% LTE 20 MHz OFDM + CFR + Hammerstein Rapp PA with memory
clear; clc; close all;

%% 0. LTE OFDM parameters (20 MHz, downlink-like)
Fs      = 30.72e6;      % sampling rate
Nfft    = 2048;         % IFFT size
NscUsed = 1200;         % active subcarriers (100 RB x 12)
SubSp   = 15e3;         % subcarrier spacing
BW_occu = NscUsed * SubSp;   % occupied BW ~= 18 MHz

Ncp     = 144;          % fixed CP length (normal CP-like)
NsymOFDM= 1000;         % number of OFDM symbols
M_qam   = 16;           % 16-QAM

fprintf('Fs = %.2f MHz, Nfft = %d, NscUsed = %d, BW_occu ≈ %.2f MHz\n',...
    Fs/1e6, Nfft, NscUsed, BW_occu/1e6);

%% 1. 16-QAM data per subcarrier
bitsPerSym_SC = log2(M_qam);
bitsPerOFDM   = bitsPerSym_SC * NscUsed;

bits  = randi([0 1], bitsPerOFDM * NsymOFDM, 1);
dataI = bi2de(reshape(bits, [], bitsPerSym_SC));       % bits -> integers
qamSym = qammod(dataI, M_qam, 'UnitAveragePower', true); % avg power = 1

qamGrid = reshape(qamSym, NscUsed, NsymOFDM);          % [NscUsed x Nsym]

%% 2. Apply a simple frequency-domain raised-cosine taper (TX shaping)
% This mimics TX filtering / windowing ⇒ roll-off at band edges.
rolloff = 0.2;                   % 20% of active band used for taper
Nedge   = floor(rolloff * NscUsed / 2);  % number of bins per edge
w = ones(NscUsed,1);

if Nedge > 0
    t = (0:Nedge-1)'/(Nedge-1);          % 0..1
    ramp = 0.5*(1 - cos(pi*t));         % raised-cosine from 0→1

    w(1:Nedge)               = ramp;        % left edge
    w(end-Nedge+1:end)       = flipud(ramp);% right edge
end

qamGrid_shaped = bsxfun(@times, qamGrid, w);

%% 3. Map onto 2048-point IFFT grid
ofdmGrid = zeros(Nfft, NsymOFDM);

scIdx = (-NscUsed/2 : NscUsed/2-1);       % -600..+599
fftBin = mod(scIdx, Nfft) + 1;            % 1..2048

ofdmGrid(fftBin, :) = qamGrid_shaped;

%% 4. IFFT + CP
tx_noCP = ifft(ofdmGrid, Nfft, 1);        % [Nfft x Nsym]
tx_withCP = [tx_noCP(end-Ncp+1:end,:); tx_noCP];   % [(Ncp+Nfft) x Nsym]

tx_bb = tx_withCP(:);                     % serialize

%% 5. Normalize average power and apply simple CFR
tx_bb = tx_bb / sqrt(mean(abs(tx_bb).^2));    % avg power = 1

papr_lin = max(abs(tx_bb).^2) / mean(abs(tx_bb).^2);
papr_dB  = 10*log10(papr_lin);
fprintf('OFDM baseband PAPR (before CFR) ≈ %.2f dB\n', papr_dB);

% Simple hard-clipping CFR to target PAPR
PAPR_target_dB = 7.5;                      % target PAPR (adjust if you like)
rms_val = sqrt(mean(abs(tx_bb).^2));
A_clip  = rms_val * 10^(PAPR_target_dB/20);

mag = abs(tx_bb);
ang = angle(tx_bb);
mag_clip = min(mag, A_clip);
x_cfr = mag_clip .* exp(1j*ang);

papr_lin_after = max(abs(x_cfr).^2) / mean(abs(x_cfr).^2);
papr_dB_after  = 10*log10(papr_lin_after);
fprintf('PAPR after CFR ≈ %.2f dB\n', papr_dB_after);

%% 6. Hammerstein Rapp PA with memory
% Memory FIR (models bias network, matching network, etc.)
h_mem = [1.0 0.1 0.01];
h_mem = h_mem / sum(abs(h_mem));      % normalize for ~1 average gain

% Rapp AM/AM parameters
A_sat  = 1.5;                         % saturation amplitude
p_rapp = 3;                           % smoothness factor
G_lin  = 4.0;                         % small-signal linear gain

% AM/PM (mild)
k_phi  = 0.3;                         % AM/PM coefficient (rad)

rappAMAM = @(r) (G_lin .* r) ./ (1 + (r./A_sat).^(2*p_rapp)).^(1/(2*p_rapp));
rappAMPM = @(r) k_phi * (r./A_sat).^2;

% Drive level (how close to compression)
drive_dB     = +1.5;                  % try 1..2 dB
drive_linear = 10^(drive_dB/20);

% PA input (after CFR)
x_in = drive_linear * x_cfr;

% Memory stage (FIR) -> x_mem is the input to the static Rapp
x_mem = filter(h_mem, 1, x_in);

% Static Rapp nonlinearity
r_in   = abs(x_mem);
phi_in = angle(x_mem);
r_out  = rappAMAM(r_in);
phi_out= phi_in + rappAMPM(r_in);
y_pa   = r_out .* exp(1j*phi_out);

fprintf('[PA only] avg|x_mem| = %.3f, peak|x_mem| = %.3f, max/Asat = %.2f\n',...
    mean(r_in), max(r_in), max(r_in)/A_sat);

%% 7. Spectra: original vs CFR vs PA output
Nfft_psd = 8192;
[pxx_tx,f]   = pwelch(tx_bb, hamming(4096), 2048, Nfft_psd, Fs, 'centered');
[pxx_cfr,~]  = pwelch(x_cfr,  hamming(4096), 2048, Nfft_psd, Fs, 'centered');
[pxx_pa, ~]  = pwelch(y_pa,   hamming(4096), 2048, Nfft_psd, Fs, 'centered');

figure;
plot(f/1e6, 10*log10(pxx_tx  + eps), 'b', 'LineWidth', 1); hold on;
plot(f/1e6, 10*log10(pxx_cfr + eps), 'm', 'LineWidth', 1);
plot(f/1e6, 10*log10(pxx_pa  + eps), 'r', 'LineWidth', 1);
grid on;
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz, normalized)');
title('Spectrum: baseband, CFR output, PA output with memory');
legend('Original baseband','After CFR','After PA (with memory)','Location','SouthWest');
xlim([-30 30]);    % look wider than just the occupied band

%% 8. Effective AM/AM (using memory input x_mem)
valid_idx = (length(h_mem):length(x_mem)).';   % drop initial FIR transient

r_in_eff  = abs(x_mem(valid_idx));
r_out_eff = abs(y_pa(valid_idx));

figure;
plot(r_in_eff, r_out_eff, '.', 'MarkerSize', 2); grid on;
xlabel('Input amplitude |x_{mem}|');
ylabel('Output amplitude |y_{PA}|');
title('Effective AM/AM: Rapp PA with memory (Hammerstein)');

%% 9. Effective AM/PM (phase of y relative to x_mem, wrapped)
phi_eff = angle(y_pa(valid_idx) .* conj(x_mem(valid_idx)));  % in [-pi, pi]

% Bin-average for a smoother curve
nbins = 60;
edges = linspace(min(r_in_eff), max(r_in_eff), nbins+1);
r_bin  = nan(nbins,1);
phi_bin= nan(nbins,1);
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
xlabel('Input amplitude |x_{mem}|');
ylabel('Phase shift (rad)');
title('Effective AM/PM: Rapp PA with memory');
legend('Scatter','Binned average','Location','SouthEast');
