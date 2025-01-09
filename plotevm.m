function [evm] = plotevm(csvfile,csvref)
data = readmatrix(csvfile);
I_data = data(:,1);
Q_data = data(:,2);
z = complex(I_data,Q_data);

csvref_txt=strrep(csvref,'.wf','.txt');
copyfile(csvref,csvref_txt)
dataref = readmatrix(csvref_txt);
Iref = dataref(:,1);
Qref = dataref(:,2);
zref = complex(Iref,Qref);
zref_norm_resample = resample(zref,1,2);

phase_offset =  angle(mean(zref_norm_resample ./z));
z_norm_aligned = z * exp(-1i * phase_offset);

plot(z_norm_aligned,'.');
hold on;
plot(zref_norm_resample,'.');
hold off;
title("phase aligned");
legend("DUT", "Ref");

symbols = length(z_norm_aligned);
nom = abs(z_norm_aligned-zref_norm_resample).^2;
denom = abs(zref_norm_resample).^2;
evm = sqrt(sum(nom./denom))/symbols;
title(["EVM%: ",evm*100])
end
