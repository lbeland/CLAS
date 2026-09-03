close all
clear all 
set(0,'defaultTextInterpreter','latex');
%%

alpha=-1;
beta=1;
gamma=0.1;
omega=1;


f1 = @(t,x,y) y;
f2 = @(t,x,y) -alpha*x-beta*x.^3 + gamma*cos(omega*t);

h=0.00001;
T=300; % reduce to 50 to speed up calculations
N=round(T/h);

t=zeros(1,N);
x=zeros(1,N);
y=zeros(1,N);

t(1)=0;
x(1)=1;
y(1)=1;

% Sol via RK4

for i=2:N
    
    [t(i),x(i),y(i)]=Euler_diff(t(i-1),x(i-1),y(i-1),h,f1,f2);

end

%% 1. 
underS = 50; 
y_proc = y(1:underS:end);
t_proc = t(1:underS:end);
Fs_new = 1/h/underS; % New sampling frequency (1000 Hz)

figure('Name','JADE Analysis: Signal', 'Units', 'Normalized', 'OuterPosition', [0 0 1 1],'DefaultAxesFontSize',36,'DefaultLineLineWidth',2);
plot(t_proc,y_proc)
title('Signal')

%% 2. FIF Decomposition

opts=Settings_FIF_v4('alpha',80,'ExtPoints',200);
L = 1/6*length(y_proc);
y_proc_e = Extend_sig_v2(y_proc,{'symw','asymw'},L,true);
IMF_y_e=FIF_v2_14(y_proc_e,opts);
IMF_y = IMF_y_e(:,L+1:end-L);
plot_imf_v10(IMF_y,t_plot,2)


%% 3. JADE IMF 1
fprintf('Processing IMF 1 with JADE...\n');
N_limit = min(160000, L);

IMF_1 = IMF_y(1,1:N_limit);
t_1 = t_plot(1:N_limit);

tic
[IFgram_1, phasegram_1, zeroCrossings_1, amplitudeGram_1] = JADE_v3(IMF_1, t_1, 1/Fs_new, 'smooth', 0, 'normalize', 0);
time_Jade_v3_IMF1=toc

fz1 = zeroCrossings_1(1); 
lz1 = zeroCrossings_1(end);
reconstr1 = amplitudeGram_1 .* cos(2*pi*phasegram_1 + pi/2);

%% 

figure
plot(amplitudeGram_1)

%% 4. JADE IMF 2
fprintf('Processing IMF 2 with JADE...\n');
up2 = 2; 
IMF_2 = IMF_y(2,1:N_limit);
t_2 = t_plot(1:N_limit);

tic
[IFgram_2, phasegram_2, zeroCrossings_2, amplitudeGram_2] = JADE_v3(IMF_2, t_2, 1/Fs_new, 'smooth', 0, 'normalize', 0);
time_Jade_v3_IMF2=toc

fz2 = zeroCrossings_2(1); 
lz2 = zeroCrossings_2(end);
reconstr2 = amplitudeGram_2 .* cos(2*pi*phasegram_2 + pi/2);

%%

figure
plot(amplitudeGram_1)

%% 5. Visualization 
figure('Name', 'JADE Analysis: Extended Range', 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);

% IMF 1 Plots
subplot(2,2,1)
t_slice1 = t_1(fz1:lz1);
len1 = min([length(t_slice1), length(reconstr1), length(IFgram_1)]);
plot(t_slice1(1:len1), IMF_1(fz1 : fz1+len1-1), 'b'); hold on;
plot(t_slice1(1:len1), reconstr1(1:len1), 'r--');
title(['IMF 1 reconstruction']); legend('Original', 'JADE'); grid on;

subplot(2,2,3)
plot(t_slice1(1:len1), IFgram_1(1:len1) * (Fs_new))
ylabel('Frequency (Hz)'); xlabel('Time (s)'); title('IF 1'); grid on;

% IMF 2 Plots
subplot(2,2,2)
t_slice2 = t_2(fz2:lz2);
len2 = min([length(t_slice2), length(reconstr2), length(IFgram_2)]);
plot(t_slice2(1:len2), IMF_2(fz2 : fz2+len2-1), 'b'); hold on;
plot(t_slice2(1:len2), reconstr2(1:len2), 'r--');
title(['IMF 2 reconstruction']); legend('Original', 'JADE'); grid on;

subplot(2,2,4)
plot(t_slice2(1:len2), IFgram_2(1:len2) * (Fs_new))
ylabel('Frequency (Hz)'); xlabel('Time (s)'); title('IF 2'); grid on;

