function [Best_rime_rate,Best_rime,Convergence_curve, Time]=RIME(N,MaxFEs,dim,A,trn,vald,TFid,classifierFhd)
tic;
if (nargin<8)
    classifierFhd = 'knn';
end
% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems
Rimepop=initialization(N,dim,1,0)>0.5;
Lb=zeros(1,dim);% lower boundary
Ub=ones(1,dim);% upper boundary
% it=1;%Number of iterations
FEs=1;

VPosition=zeros(1,dim);
VFitness=0;
Limits=50;
SalpTrial=zeros(1,N);
S=zeros(N,N-1);
F1=1.0;
Cr1=0.1;
U1Positions=zeros(1,dim);
U1Fitness=0;
F2=0.8;
Cr2=0.2;
U2Positions=zeros(1,dim);
U2Fitness=0;
F3=1.0;
Cr3=0.9;
U3Positions=zeros(1,dim);
U3Fitness=0;

aa=10;
bb=0.5;
Convergence_curve=[];
Rime_rates=zeros(1,N);%Initialize the fitness value
newRime_rates=zeros(1,N);
W = 5;%Soft-rime parameters, discussed in subsection 4.3.1 of the paper
%Calculate the fitness value of the initial position
for i=1:N
    Rime_rates(1,i)=AccSz2(Rimepop(i,:), A,trn,vald,classifierFhd);%Calculate the fitness value for each search agent
    %Make greedy selections
    if Rime_rates(1,i)<Best_rime_rate
        Best_rime_rate=Rime_rates(1,i);
        Best_rime=Rimepop(i,:);
    end
end
% Main loop
while FEs <= MaxFEs
    RimeFactor = (rand-0.5)*2*cos((pi*FEs/(MaxFEs/10)))*(1-round(FEs*W/MaxFEs)/W);%Parameters of Eq.(3),(4),(5)
    E =sqrt(FEs/MaxFEs);%Eq.(6)
    newRimepop = double(Rimepop);%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    c=2*(1-(FEs/MaxFEs));
    for i=1:N
        Escaping_Energy=c*(2*rand()-1);  % escaping energy of rabbit
        if abs(Escaping_Energy)>=1
            r=rand();
            rand_Hawk_index = floor(N*rand()+1);
            X_rand = newRimepop(rand_Hawk_index, :);
            if r<0.5
                % perch based on other family members
                newRimepop(i,:)=X_rand-rand()*abs(X_rand-2*rand()*newRimepop(i,:));
            elseif r>0.5
                % perch on a random tall tree (random site inside group's home range)
                newRimepop(i,:)=(Best_rime(1,:)-mean(newRimepop))-rand()*((1-0)*rand+0);
            end
            
        else
            for j=1:dim
                %Soft-rime search strategy
                r1=rand();
                if r1< E
                    newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)
                end
                %Hard-rime puncture mechanism
                r2=rand();
                if r2<normalized_rime_rates(i)
                    newRimepop(i,j)=Best_rime(1,j);%Eq.(7)
                end
            end
        end
    end
    for i=1:N
        %Boundary absorption
        for jj=1:dim
            xx=sigmf(newRimepop(i,jj),[aa bb]);
%                         xx=(1-exp(-2*newRimepop(i,jj)))/(1+exp(-2*newRimepop(i,jj)));
            
            if xx<rand
                xx=0;
            else
                xx=1;
            end
            newRimepop(i,jj)=xx;% Equation (3.7)
        end
        
        
        newRime_rates(1,i)=AccSz2(newRimepop(i,:), A,trn,vald,classifierFhd);
        
        %Positive greedy selection mechanism
        if newRime_rates(1,i)<Rime_rates(1,i)
            Rime_rates(1,i) = newRime_rates(1,i);
            Rimepop(i,:) = newRimepop(i,:);
            if newRime_rates(1,i)< Best_rime_rate
                Best_rime_rate=Rime_rates(1,i);
                Best_rime=Rimepop(i,:);
            end
            SalpTrial(1,i)=0;
        else
            SalpTrial(1,i)=SalpTrial(1,i)+1;
        end
        S(i,:)=[1:i-1,i+1:N];
        
        K1=randperm(N-1,3);
        for j=1:dim
            if rand()<Cr1 || j==randperm(dim,1)
                U1Positions(j)=Rimepop(S(i,K1(1)),j)+F1*(Rimepop(S(i,K1(2)),j)-Rimepop(S(i,K1(3)),j));
            else
                U1Positions(j)=Rimepop(i,j);
            end
        end
        
        
        for jj=1:dim
            xx=sigmf(U1Positions(jj),[aa bb]);
%                         xx=(1-exp(-2*U1Positions(jj)))/(1+exp(-2*U1Positions(jj)));
            
            if xx<rand
                xx=0;
            else
                xx=1;
            end
            U1Positions(jj)=xx;% Equation (3.7)
        end
        
        U1Fitness=AccSz2(U1Positions, A,trn,vald,classifierFhd);
        
        
        K2=randperm(N-1,5);
        for j=1:dim
            if rand()<Cr2 || j==randperm(dim,1)
                U2Positions(j)=Rimepop(S(i,K2(1)),j)+F2*(Rimepop(S(i,K2(2)),j)-Rimepop(S(i,K2(3)),j))+F2*(Rimepop(S(i,K2(4)),j)-Rimepop(S(i,K2(5)),j));
            else
                U2Positions(j)=Rimepop(i,j);
            end
        end
        for jj=1:dim
            xx=sigmf(U2Positions(jj),[aa bb]);
%                         xx=(1-exp(-2*U2Positions(jj)))/(1+exp(-2*U2Positions(jj)));
            
            if xx<rand
                xx=0;
            else
                xx=1;
            end
            U2Positions(jj)=xx;% Equation (3.7)
        end
        U2Fitness=AccSz2(U2Positions, A,trn,vald,classifierFhd);
        
        K3=randperm(N-1,3);
        for j=1:dim
            if rand()<Cr3 || j==randperm(dim,1)
                U3Positions(j)=Rimepop(i,j)+rand()*(Rimepop(S(i,K3(1)),j)-Rimepop(i,j))+F3*(Rimepop(S(i,K3(2)),j)-Rimepop(S(i,K3(3)),j));
            else
                U3Positions(j)=Rimepop(i,j);
            end
        end
        for jj=1:dim
            xx=sigmf(U3Positions(jj),[aa bb]);
%                         xx=(1-exp(-2*U3Positions(jj)))/(1+exp(-2*U3Positions(jj)));
            
            if xx<rand
                xx=0;
            else
                xx=1;
            end
            U3Positions(jj)=xx;% Equation (3.7)
        end
        U3Fitness=AccSz2(U3Positions, A,trn,vald,classifierFhd);
        
        ComFitness=[U1Fitness,U2Fitness,U3Fitness];
        [~,MinIndex]=min(ComFitness);
        switch MinIndex
            case 1
                VPosition=U1Positions;
                VFitness=U1Fitness;
            case 2
                VPosition=U2Positions;
                VFitness=U2Fitness;
            case 3
                VPosition=U3Positions;
                VFitness=U3Fitness;
        end
        if VFitness<Rime_rates(1,i)
            Rimepop(i,:)=VPosition;
            Rime_rates(1,i)=VFitness;
            SalpTrial(1,i)=0;
        else
            SalpTrial(1,i)=SalpTrial(1,i)+1;
        end
        if SalpTrial(1,i)>=Limits
            T1=0+rand(1,dim).*(1-0);
            T2=rand(1,dim).*(1+0)-Rimepop(i,:);
            for jj=1:dim
                xx=sigmf(T1(jj),[aa bb]);
%                             xx=(1-exp(-2*T1(jj)))/(1+exp(-2*T1(jj)));
                
                if xx<rand
                    xx=0;
                else
                    xx=1;
                end
                T1(jj)=xx;% Equation (3.7)
            end
            for jj=1:dim
                xx=sigmf(T2(jj),[aa bb]);
%                             xx=(1-exp(-2*T2(jj)))/(1+exp(-2*T2(jj)));
                
                if xx<rand
                    xx=0;
                else
                    xx=1;
                end
                T2(jj)=xx;% Equation (3.7)
            end
            
            
            fT1=AccSz2(T1, A,trn,vald,classifierFhd);
            fT2=AccSz2(T2, A,trn,vald,classifierFhd);
            
            if fT1<fT2
                Rimepop(i,:)=T1;
                Rime_rates(1,i)=fT1;
            else
                Rimepop(i,:)=T2;
                Rime_rates(1,i)=fT2;
            end
            SalpTrial(1,i)=0;
        end
        if Rime_rates(1,i)<Best_rime_rate
            Best_rime=Rimepop(i,:);
            Best_rime_rate=Rime_rates(1,i);
        end
    end
    
    
    
    
    
    
    Convergence_curve(FEs)=Best_rime_rate;
    FEs=FEs+1;
end
Time = toc;
