%% Script A: LTE 20 MHz 16QAM OFDM baseband
clear; clc; close all;

%% ----------------- LTE OFDM Parameters -----------------
Fs      = 30.72e6;      % sampling rate for LTE 20 MHz
Nfft    = 2048;         % IFFT size
NscUsed = 1200;         % number of active subcarriers (100 RB × 12)
SubSp   = 15e3;         % subcarrier spacing
BW_occu = NscUsed * SubSp;  % occupied BW ≈ 18 MHz

Ncp     = 144;          % simplified constant CP (typical normal CP)
NsymOFDM= 1000;         % number of OFDM symbols in time
M_qam   = 16;           % 16QAM per subcarrier

fprintf('Fs = %.2f MHz, Nfft = %d, NscUsed = %d, occupied BW ≈ %.2f MHz\n', ...
    Fs/1e6, Nfft, NscUsed, BW_occu/1e6);

%% ----------------- 1. 16QAM -----------------
bitsPerSym_SC = log2(M_qam);        % bits per subcarrier
bitsPerOFDM   = bitsPerSym_SC * NscUsed;

bits = randi([0 1], bitsPerOFDM * NsymOFDM, 1);

% Group every log2(M) bits into one integer, then perform 16QAM modulation
dataInt = bi2de(reshape(bits, [], bitsPerSym_SC));
qamSym  = qammod(dataInt, M_qam, 'UnitAveragePower', true);  % avg power = 1

% Reshape the resulting symbols into an array of dimensions [Subcarriers x OFDM Symbols]
qamGrid = reshape(qamSym, NscUsed, NsymOFDM);

%% ----------------- 2. Map the frequency-domain symbols onto the 2048 IFFT grid (Inverse Fast Fourier Transform grid size) -----------------
% LTE 20 MHz Configuration: Place 1200 consecutive subcarriers at the center of the spectrum (including both sides of the DC (Direct Current) carrier)
ofdmGrid = zeros(Nfft, NsymOFDM);

% Frequency Domain Indexing: Use center-symmetric indexing, mapping indices -600 through +599 onto the IFFT grid
scIdx = (-NscUsed/2 : NscUsed/2-1);          % -600 ... +599
fftBin = mod(scIdx, Nfft) + 1;               % Translate to index 1..2048 

ofdmGrid(fftBin, :) = qamGrid;

%% ----------------- 3. 2048 IFFT -> Time Domain OFDM -----------------
tx_noCP = ifft(ofdmGrid, Nfft, 1);           % [Nfft × NsymOFDM]

%% ----------------- 4. Add the Cyclic Prefix (CP) -----------------
tx_withCP = [tx_noCP(end-Ncp+1:end, :); tx_noCP];   % [(Ncp+Nfft) × NsymOFDM]

% Serialize the data into a one-dimensional time-domain baseband sequence
tx_bb = tx_withCP(:);

%% ----------------- 5. Power normalization & PAPR (Peak-to-Average Power Ratio) adjustment/calculation -----------------
tx_bb = tx_bb / sqrt(mean(abs(tx_bb).^2));  % Normalize the average power to 1

papr_lin = max(abs(tx_bb).^2) / mean(abs(tx_bb).^2);
papr_dB  = 10*log10(papr_lin);
fprintf('OFDM baseband PAPR ≈ %.2f dB\n', papr_dB);

%% ----------------- 6. Plot the spectrum (it should be flat in the center, with a bandwidth of approximately 18MHz) -----------------
Nfft_psd = 8192;
[pxx,f]  = pwelch(tx_bb, hamming(4096), 2048, Nfft_psd, Fs, 'centered');

figure;
plot(f/1e6, 10*log10(pxx + eps), 'LineWidth', 1);
grid on;
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz, normalized)');
title('LTE 20 MHz 16QAM OFDM Baseband Spectrum');
xlim([-15 15]);

%% ----------------- 7. Examine the constellation on a single subcarrier -----------------
% To visually inspect the 16QAM, arbitrarily select a subcarrier (e.g., the one in the middle) and perform FFT demodulation
symLen   = Ncp + Nfft;
Nsym_rx  = floor(length(tx_bb) / symLen);
rx_mat   = reshape(tx_bb(1:Nsym_rx*symLen), symLen, Nsym_rx);

% Remove CP
rx_noCP = rx_mat(Ncp+1:end, :);

% FFT
RxGrid = fft(rx_noCP, Nfft, 1);   % [Nfft × Nsym_rx]

% Use the same subcarrier index as was used during transmission
RxUsed = RxGrid(fftBin, :);

% Select a central subcarrier, for example, index = 600 (near 0Hz)
scPick = floor(NscUsed/2) + 1;    % The central subcarrier
pickedSym = RxUsed(scPick, :);   % all 16QAM symbols within the time slot on this specific subcarrier

figure;
plot(real(pickedSym), imag(pickedSym), '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('One subcarrier constellation');
