generate_barcode;

S = length(patterns);

patternLengths = zeros(S,1);
for s = 1:S
    patternLengths(s) = length(patterns{s});
end

M = 6;
C = max(patternLengths);
NumStates = C*S*M;

%Enumerate all the states
States = zeros(NumStates,3) -1 ;
StatesInv = zeros(C,S,M) -1;

ix = 1;
for c = 1:C
    for s = 1:S
        for m = 1:M
            if( (s > 5) || (s <=5 && m ==1) )
                if( c<=patternLengths(s))
                    States(ix,:) = [c s m];
                    StatesInv(c,s,m) = ix;
                    ix = ix+1;
                end
            end
        end
    end
end

NumStates = ix -1;
States = States(1:NumStates,:);

%warning: NumStates will be less than S*M*C, because not all possible
%[s,m,c] triples are valid. 


%% Part1: Fill the transition matrix A

%mapping states to binary numbers, which will be useful for computing the
%likelihood
f_kst = zeros(NumStates,1); 


A = zeros(NumStates);

for i = 1:NumStates
    
    c = States(i,1);
    s = States(i,2);
    m = States(i,3);
    
    patternLen = patternLengths(s);
    f_kst(i) = patterns{s}(c); %determines if this state is black or white
    
    %example:
    if(s == 1) %starting quiet zone
        
        if(c == patternLen)
            
            for ss = [1 3] %the next states can only be either starting quiet zone, or the starting guard
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = (1/2);
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
        
    elseif(s == 2) %ending quiet zone
        if(c == patternLen)
            % the scaning of the barcode finished
            for ss = 2 %the next states can only be ending quiet zone
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        %to be filled
        
    elseif(s== 3) %starting guard
        if(c == patternLen)
            
            for ss = 6:15 %the next states can only be the left digits
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = (1/10);
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        %to be filled
        
    elseif(s== 4) %ending guard
        if(c == patternLen)
            
            for ss = 2 %the next states can only be the ending quiet zone
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        %to be filled
        
    elseif(s== 5) %middle guard
        if(c == patternLen)
            
            for ss = 16:25 %the next states can only be the right digits
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = (1/10);
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        %to be filled
        
    elseif(s>= 6 && s<=15) %left symbols
        if(c == patternLen)%the next states can only be either the left digits or middle code
            if(m <6)
                for ss = 6:15 

                    s_next = ss;
                    c_next = 1;
                    m_next = m+1;

                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = (1/10);               
                    
                end
            else
                s_next = 5;
                c_next = 1;
                m_next = 1;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        %to be filled
            
    elseif(s>= 16 && s<=25) %right symbols
        if(c == patternLen) %the next states can only be either the right digits or ending guard
            if(m <6)
                for ss = 16:25 

                    s_next = ss;
                    c_next = 1;
                    m_next = m+1;

                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = (1/10);               
                    
                end
            else
                s_next = 4;
                c_next = 1;
                m_next = 1;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        %to be filled
        
    else
        error('Unknown State!');
    end
end

%% Part2: Compute the inital probability
p_init = zeros(NumStates,1);
ix = StatesInv(1,1,1);
p_init(ix) = 1;

%the barcode *must* start with the "starting quite zone", with s_n=1. Other
%states are not possible. Fill the initial probability accordingly. 

%% Q3 simulate the HMM and visualize the simulated data.
mu= [250 20]';
sigma = [sqrt(5) sqrt(5)]';
N = 125;
rand_state = RandStream('mlfg6331_64');
% generate original state according to p_init
s0 = datasample(rand_state, 1:NumStates,1,'Weights',p_init);
simu_states = zeros(N, 3);
simu_x = zeros(N,1);
isn = 0;
for i = 1:N
    if(i==1)
%         c = States(s0,1);
%         s = States(s0,2);
%         m = States(s0,3);
%         ix = StatesInv(c,s,m);
        isn = datasample(rand_state,1:NumStates,1,'Weights',p_init);
        simu_states(i,:) = States(isn,:);
    else
        c = simu_states(i-1,1);
        s = simu_states(i-1,2);
        m = simu_states(i-1,3); 
        ix = StatesInv(c,s,m);
        isn = datasample(rand_state,1:NumStates,1,'Weights',A(:,ix));
        simu_states(i,:) = States(isn,:);
    end
    if(f_kst(isn)==0)
        simu_x(i) = normrnd(mu(1),sigma(1));
    else
        simu_x(i) = normrnd(mu(2),sigma(2));
    end
    
end

figure;
plot(simu_x, '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_n$', 'Interpreter', 'latex')
bc_image_simu = uint8((repmat(simu_x', [100 1])));
figure;
imshow(bc_image_simu)
set(gcf,'Position',[100 100 1000 500]);
title(num2str(code));
figure;
plot(simu_states(:,1), '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$c_n$', 'Interpreter', 'latex');
figure;
plot(simu_states(:,2), '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$s_n$', 'Interpreter', 'latex');
figure;
plot(simu_states(:,3), '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$m_n$', 'Interpreter', 'latex');
%% Part3: Compute the log-likelihood

T = length(obs);

logObs = zeros(NumStates,T);

% mu= [mu0 mu1]';
% sigma = ...;
mu= [255 0]';
sigma = [1 1]';

for t=1:T
    % to be filled
    % you can use the variable f_kst here
    
    % white space
    logObs(f_kst==0,t) = log(normpdf(255-obs(t),mu(1),sigma(1)) + realmin); % realmin: avoid inf
    % black space
    logObs(f_kst==1,t) = log(normpdf(255-obs(t),mu(2),sigma(2)) + realmin);
end

%% Part 4: Compute the filtering distribution via Forward recursion

alphakk = zeros(NumStates, T);
alphakk1 = zeros(NumStates, T);

for i = 1:T
    if(i==1)
        alphakk1(:,i) = log(p_init); 
        alphakk(:,i) = logObs(:,i)+alphakk1(:,i);
    else
        alphakk1(:,i) = log(A*exp(alphakk(:,i-1)-max(alphakk(:,i-1)))) + max(alphakk(:,i-1));
        alphakk(:,i) = logObs(:,i)+alphakk1(:,i);
    end
end

log_fDistri = zeros(NumStates,T);
for t=1:T
    mx=max(alphakk(:,t));
    logsum = log(sum(exp(alphakk(:,t)-mx)))+mx;
    log_fDistri(:,t)=alphakk(:,t)-logsum;
end

fDistri = exp(log_fDistri);

%%
c_fd = zeros(C,T);
for i = 1:C
    c_fd(i,:) = sum(fDistri(States(:,1)==i,:));
end
s_fd = zeros(S,T);
for i = 1:S
    s_fd(i,:) = sum(fDistri(States(:,2)==i,:));
end
m_fd = zeros(M,T);
for i = 1:M
    m_fd(i,:) = sum(fDistri(States(:,3)==i,:));
end
figure;
subplot(2,2,1);
imagesc(fDistri);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$\Psi_k (state)$','interpreter','latex');
title('the filtering distribution of $\Psi_k$','interpreter', 'latex');
caxis([0 1]);
colorbar
subplot(2,2,2);
imagesc(c_fd);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$c_k$', 'interpreter','latex');
title('the filtering distribution of $c_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,3);
imagesc(s_fd);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$s_k$', 'interpreter','latex');
title('the filtering distribution of $s_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,4);
imagesc(m_fd);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$m_k$','interpreter','latex');
title('the filtering distribution of $m_k$','interpreter', 'latex');
caxis([0 1]);
colorbar
%% Part 5: Compute the smoothing distribution via Forward-Backward recursion
betakk = zeros(NumStates, T);
betakk1 = zeros(NumStates, T);

for i = T:-1:1
   if(i==T)
      betakk1(:, i) = 0;
      betakk(:, i) = logObs(:, i) + betakk1(:, i);
   else
      betakk1(:, i) = log(A'*exp(betakk(:,i+1)-max(betakk(:,i+1)))) + max(betakk(:,i+1));
      betakk(:, i) = logObs(:, i) + betakk1(:, i); 
   end   
end

loggammak = zeros(NumStates,T);
log_sDistri = zeros(NumStates,T);   
for t=1:T
    loggammak(:,t)=betakk1(:,t)+alphakk(:,t);
    mx = max(loggammak(:,t));
    logsum = log(sum(exp(loggammak(:,t)- mx)))+ mx;
    log_sDistri(:,t) = loggammak(:,t) - logsum;
end

sDistri = exp(log_sDistri);

%% 
c_sd = zeros(C,T);
for i = 1:C
    c_sd(i,:) = sum(sDistri(States(:,1)==i,:));
end

s_sd = zeros(S,T);
for i = 1:S
    s_sd(i,:) = sum(sDistri(States(:,2)==i,:));
end

m_sd = zeros(M,T);
for i = 1:M
    m_sd(i,:) = sum(sDistri(States(:,3)==i,:));    
end
figure;
subplot(2,2,1);
imagesc(sDistri);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$\Psi_k (state)$','interpreter','latex');
title('the smoothing distribution of $\Psi_k$','interpreter', 'latex');
caxis([0 1]);
colorbar
% plot(log_sum_exp(log_gamma, 1));
subplot(2,2,2);
imagesc(c_sd);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$c_k$', 'interpreter','latex');
title('the smoothing distribution of $c_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,3);
imagesc(s_sd);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$s_k$', 'interpreter','latex');
title('the smoothing distribution of $s_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,4);
imagesc(m_sd);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$m_k$','interpreter','latex');
title('the smoothing distribution of $m_k$','interpreter', 'latex');
caxis([0 1]);
colorbar
%% Part 6: Compute the most-likely path via Viterbi algorithm
viterbi_path = zeros(NumStates,T);
logA = log(A);
for i = 1:T
    if(i==1)
        V = log(p_init) + logObs(:,i);
        [V, viterbi_path(:,i)] = max(logA+V',[],2);
    elseif(i<T)
        V = logObs(:,i)+V;
        [V, viterbi_path(:,i)] = max(logA+V',[],2);
    elseif(i==T)
        viterbi_path(:,i) = logObs(:,i)+V;
    end
end

best_path = zeros(T,1);
for i = T:-1:1
    if(i==T)
      [~, best_path(i)] = max(viterbi_path(:,i));
    else
        best_path(i) = viterbi_path(best_path(i+1),i);
    end
end

best_states = zeros(T,3);
best_state_path = zeros(NumStates,T);
best_c = zeros(C,T);
best_s = zeros(S,T);
best_m = zeros(M,T);
for i = 1:T
   best_states(i,:) = States(best_path(i),:); 
   c = best_states(i,1);
   s = best_states(i,2);
   m = best_states(i,3);
   best_state_path(best_path(i),i) = 1;
   best_c(c,i) = 1;
   best_s(s,i) = 1;
   best_m(m,i) = 1;
end

%to be filled
%%
figure;
subplot(2,2,1);
imagesc(best_state_path);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$\Psi_k$', 'interpreter','latex');
title('the best path of $\Psi_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,2);
imagesc(best_c);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$c_k$', 'interpreter','latex');
title('the best path of $c_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,3);
imagesc(best_s);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$s_k$', 'interpreter','latex');
title('the best path of $s_k$','interpreter', 'latex');
caxis([0 1]);
colorbar

subplot(2,2,4);
imagesc(best_m);
set(gca, 'ydir', 'n');
colormap(flipud(gray));
xlabel('k (time)'); ylabel('$m_k$','interpreter','latex');
title('the best path of $m_k$','interpreter', 'latex');
caxis([0 1]);
colorbar


%% Part 7: Obtain the barcode string from the decoded states

best_cn = best_states(:,1); % (this will be obtained via Viterbi)
best_sn = best_states(:,2); %(this will be obtained via Viterbi)

%find the place where a new symbol starts
ix = find(best_cn ==1);

s_ix = best_states(ix,2);
decoded_code = [];
for i = 1:length(s_ix)
    tmp = s_ix(i);
    %consider only the symbols that correspond to digits
    if(tmp>=6)
        chr = mod(tmp-6,10);
        decoded_code = [decoded_code, chr];
    end   
end

fprintf('Real code:\t');
fprintf('%d',code);
fprintf('\n');
fprintf('Decoded code:\t');
fprintf('%d',decoded_code);
fprintf('\n');
