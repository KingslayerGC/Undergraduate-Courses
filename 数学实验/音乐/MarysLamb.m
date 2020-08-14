%% Matlab°æ¡¶ÂêÀöµÄÐ¡Ñò¸á¡·
% Coding by Recksic
%% define the basic sound wave
fs = 44100; % Standard sample rate
dt = 1/fs; % Standard sampling time interval
T16 = 1/fs*3199; %To determine the time length of a  1/16 note, suggest as an odd number
t16 = [0:dt:T16];
[temp k] = size(t16);
t1 = linspace(0,16*T16,16*k);% An array with the same length as a full note
t2 = linspace(0,8*T16,8*k);
t4 = linspace(0,4*T16,4*k);
t4d = linspace(0,6*T16,6*k);%A special array represents a 1/4+1/8 note
t8 = linspace(0,2*T16,2*k);
mod1 = sin(pi*t1/t1(end));% Defining a basic amplitude function (So that the sound won't suddenly occur or vanish
mod2 = sin(pi*t2/t2(end));
mod4 = sin(pi*t4/t4(end));
mod4d = sin(pi*t4d/t4d(end));
mod8 = sin(pi*t8/t8(end));


%% Frequency and note List
f0 = 2^(1/4)*261.6; % 1 = E^b , which is three half tones higher than C tone
ScaleTable = [1 2^(2/12) 2^(4/12) 2^(5/12) 2^(7/12) 2^(9/12) 2^(11/12)];%Other frequencies

           % full notes
do2F = mod1.*cos(2*pi*ScaleTable(1)*f0*t1);
re2F = mod1.*cos(2*pi*ScaleTable(2)*f0*t1);
mi2F = mod1.*cos(2*pi*ScaleTable(3)*f0*t1);
fa2F = mod1.*cos(2*pi*ScaleTable(4)*f0*t1);
so2F = mod1.*cos(2*pi*ScaleTable(5)*f0*t1);
la2F = mod1.*cos(2*pi*ScaleTable(6)*f0*t1);
xi2F = mod1.*cos(2*pi*ScaleTable(7)*f0*t1);
blkF = zeros(size(mod1));
          % 1/2 notes
do2h = mod2.*cos(2*pi*ScaleTable(1)*f0*t2);
re2h = mod2.*cos(2*pi*ScaleTable(2)*f0*t2);
mi2h = mod2.*cos(2*pi*ScaleTable(3)*f0*t2);
fa2h = mod2.*cos(2*pi*ScaleTable(4)*f0*t2);
so2h = mod2.*cos(2*pi*ScaleTable(5)*f0*t2);
la2h = mod2.*cos(2*pi*ScaleTable(6)*f0*t2);
xi2h = mod2.*cos(2*pi*ScaleTable(7)*f0*t2);
blkh = zeros(size(mod2));
% 1/4+1/2 notes
do2fd = mod4d.*cos(2*pi*ScaleTable(1)*f0*t4d);
re2fd = mod4d.*cos(2*pi*ScaleTable(2)*f0*t4d);
mi2fd = mod4d.*cos(2*pi*ScaleTable(3)*f0*t4d);
fa2fd = mod4d.*cos(2*pi*ScaleTable(4)*f0*t4d);
so2fd = mod4d.*cos(2*pi*ScaleTable(5)*f0*t4d);
la2fd = mod4d.*cos(2*pi*ScaleTable(6)*f0*t4d);
xi2fd = mod4d.*cos(2*pi*ScaleTable(7)*f0*t4d);
blkfd = zeros(size(mod4d));
% 1/4 notes
do2f = mod4.*cos(2*pi*ScaleTable(1)*f0*t4);
re2f = mod4.*cos(2*pi*ScaleTable(2)*f0*t4);
mi2f = mod4.*cos(2*pi*ScaleTable(3)*f0*t4);
fa2f = mod4.*cos(2*pi*ScaleTable(4)*f0*t4);
so2f = mod4.*cos(2*pi*ScaleTable(5)*f0*t4);
la2f = mod4.*cos(2*pi*ScaleTable(6)*f0*t4);
xi2f = mod4.*cos(2*pi*ScaleTable(7)*f0*t4);
blkf = zeros(size(mod4));
% 1/8 notes
do2e = mod8.*cos(2*pi*ScaleTable(1)*f0*t8);
re2e = mod8.*cos(2*pi*ScaleTable(2)*f0*t8);
mi2e = mod8.*cos(2*pi*ScaleTable(3)*f0*t8);
fa2e = mod8.*cos(2*pi*ScaleTable(4)*f0*t8);
so2e = mod8.*cos(2*pi*ScaleTable(5)*f0*t8);
la2e = mod8.*cos(2*pi*ScaleTable(6)*f0*t8);
xi2e = mod8.*cos(2*pi*ScaleTable(7)*f0*t8);
blke = zeros(size(mod8));

%% Melody     
s = [mi2fd re2e do2f re2f mi2f mi2f mi2h, re2f re2f re2h, mi2f mi2f+so2f mi2h+so2h,...
    mi2fd re2e do2f re2f mi2f mi2f mi2f do2e do2e re2f re2f mi2f re2f do2F];
       
%% Processes before play
s = s/max(s); %Balance the amplitude never greater than 1
% audiowrite('MarysLamb.mp4',s,fs);%Save the music to a file
% audiowrite('MarysLamb.flac',s,fs);%Save the music to a file
sound(s,fs);%Play the sound