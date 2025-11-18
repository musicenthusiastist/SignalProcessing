%% Script A: LTE 20 MHz 16QAM OFDM baseband
clear; clc; close all;

%% ----------------- LTE OFDM 基本参数 -----------------
Fs      = 30.72e6;      % sampling rate for LTE 20 MHz
Nfft    = 2048;         % IFFT size
NscUsed = 1200;         % number of active subcarriers (100 RB × 12)
SubSp   = 15e3;         % subcarrier spacing
BW_occu = NscUsed * SubSp;  % occupied BW ≈ 18 MHz

Ncp     = 144;          % simplified constant CP (normal CP 里典型值)
NsymOFDM= 1000;         % number of OFDM symbols in time
M_qam   = 16;           % 16QAM per subcarrier

fprintf('Fs = %.2f MHz, Nfft = %d, NscUsed = %d, occupied BW ≈ %.2f MHz\n', ...
    Fs/1e6, Nfft, NscUsed, BW_occu/1e6);

%% ----------------- 1. 随机比特 -> 16QAM 符号 -----------------
bitsPerSym_SC = log2(M_qam);        % bits per subcarrier
bitsPerOFDM   = bitsPerSym_SC * NscUsed;

bits = randi([0 1], bitsPerOFDM * NsymOFDM, 1);

% 按每 (log2 M) 个 bit 组成一个整数，再 16QAM 调制
dataInt = bi2de(reshape(bits, [], bitsPerSym_SC));
qamSym  = qammod(dataInt, M_qam, 'UnitAveragePower', true);  % avg power = 1

% reshape 成 [子载波 × OFDM 符号]
qamGrid = reshape(qamSym, NscUsed, NsymOFDM);

%% ----------------- 2. 频域映射到 2048 IFFT 栅格 -----------------
% LTE 20 MHz：把 1200 个连续子载波放在频谱中心（包含 DC 两侧）
ofdmGrid = zeros(Nfft, NsymOFDM);

% 频域索引：中心对称，-600..+599 映到 IFFT 栅格
scIdx = (-NscUsed/2 : NscUsed/2-1);          % -600 ... +599
fftBin = mod(scIdx, Nfft) + 1;               % 转成 1..2048 的索引

ofdmGrid(fftBin, :) = qamGrid;

%% ----------------- 3. 2048 IFFT -> 时域 OFDM -----------------
tx_noCP = ifft(ofdmGrid, Nfft, 1);           % [Nfft × NsymOFDM]

%% ----------------- 4. 加 CP -----------------
tx_withCP = [tx_noCP(end-Ncp+1:end, :); tx_noCP];   % [(Ncp+Nfft) × NsymOFDM]

% 串成一维时域基带序列
tx_bb = tx_withCP(:);

%% ----------------- 5. 功率归一化 & PAPR -----------------
tx_bb = tx_bb / sqrt(mean(abs(tx_bb).^2));  % 归一化平均功率为 1

papr_lin = max(abs(tx_bb).^2) / mean(abs(tx_bb).^2);
papr_dB  = 10*log10(papr_lin);
fprintf('OFDM baseband PAPR ≈ %.2f dB\n', papr_dB);

%% ----------------- 6. 画频谱（应该是中间平、带宽 ≈18 MHz） -----------------
Nfft_psd = 8192;
[pxx,f]  = pwelch(tx_bb, hamming(4096), 2048, Nfft_psd, Fs, 'centered');

figure;
plot(f/1e6, 10*log10(pxx + eps), 'LineWidth', 1);
grid on;
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz, normalized)');
title('LTE 20 MHz 16QAM OFDM Baseband Spectrum');
xlim([-15 15]);

%% ----------------- 7. 看一个子载波上的“星座” -----------------
% 为了直观看到 16QAM，我们随便挑一个子载波（比如中间那个）做 FFT 解调
symLen   = Ncp + Nfft;
Nsym_rx  = floor(length(tx_bb) / symLen);
rx_mat   = reshape(tx_bb(1:Nsym_rx*symLen), symLen, Nsym_rx);

% 去 CP
rx_noCP = rx_mat(Ncp+1:end, :);

% FFT
RxGrid = fft(rx_noCP, Nfft, 1);   % [Nfft × Nsym_rx]

% 取和发送时相同的子载波索引
RxUsed = RxGrid(fftBin, :);

% 选中间一个子载波，比如 index = 600（0 Hz 附近）
scPick = floor(NscUsed/2) + 1;    % 中心子载波
pickedSym = RxUsed(scPick, :);   % 这一条子载波上的时隙内所有 16QAM 符号

figure;
plot(real(pickedSym), imag(pickedSym), '.');
axis square; grid on;
xlabel('I'); ylabel('Q');
title('One subcarrier constellation (should look like 16QAM)');
