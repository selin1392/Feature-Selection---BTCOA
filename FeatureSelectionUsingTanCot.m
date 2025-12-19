% A new metaheuristic algorithm:
% TAN COT Optimaizer (BTCOA) 

function Out = FeatureSelectionUsingTanCot(Params,TrainData)

disp('Features Selected with wrapper approach (Starting BTCOA)');
%% Problem Definition
CostFunction = @(s) EvaluateFeatures(s,TrainData); % Cost Function

nVar = size(TrainData.Inputs,2); % Number of Decision Variables
VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMin = -3;         % Lower Bound of Variables
VarMax = 3;         % Upper Bound of Variables

VarRange=[VarMin VarMax];   % Variation Range of Variables

VelMax=(VarMax-VarMin);  % Maximum Velocity
VelMin=VarMin;



%% BTCOA Parameters

MaxIt=Params.MaxIt;      % Maximum Number of Iterations

nPop=Params.nPop;        % Population Size

% bc=unifrnd(0.52,0.60);

 bc=unifrnd(0.6,0.8);

tic

%% Initialization
% Empty Structure to Hold Individuals Data
empty_individual.Position=[];
empty_individual.Velocity=[];
empty_individual.Cost=[];
empty_individual.Best.Position=[];
empty_individual.Best.Cost=[];

empty_individual.S = [];
empty_individual.Out=[];

% Create Population Matrix
pop=repmat(empty_individual,nPop,1);

BestSol.Cost=inf;
BestSol.out=[];
G1=BestSol;

% Initialize Position
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Velocity=zeros(VarSize);


    % ABS COS transfer (Binary Coding)
      pop(i).S = abs(cos(pop(i).Position)) > bc;

  %% 
    
    [pop(i).Cost, pop(i).Out] = CostFunction(pop(i).S);

    G2=pop(i);
    if G2.Cost<G1.Cost
        G1= G2;
        BestSol=G1;
    end


end

% Vector to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% Main loop
for It = 1:MaxIt

    for i = 1:nPop
        a=-1+It*((-1)/MaxIt);
        alpha=(a-1)*rand+1;

        c =unifrnd(0.1,0.4);
        r1=c-It*((c)/MaxIt);

        d=unifrnd(1,2);
        w1=d-It*((d)/MaxIt);

        g=unifrnd(0.3,0.4);
        r=g+0.3*(It/MaxIt);


        gamma= unifrnd(3,7)*pi;
        b=unifrnd(-0.1,0.6);

        E1= exp((-b).*gamma).*tan(alpha.*gamma);
        D1=(G1.Position-pop(i).Position);
        if rand()<r
            pop(i).Velocity =E1*(r1.*pop(i).Velocity +w1.*D1);

            % Apply Velocity Bounds
            pop(i).Velocity=min(max(pop(i).Velocity,VelMin),VelMax);

            % Update Position
            pop(i).Position=pop(i).Position+pop(i).Velocity;

            % Velocity Reflection
            flag=(pop(i).Position<VarMin | pop(i).Position>VarMax);
            pop(i).Velocity(flag)=-pop(i).Velocity(flag);

            % Apply Position Bounds
            pop(i).Position=min(max(pop(i).Position,VarMin),VarMax);

            % Arctan transfer (Binary Coding)
            % pop(i).S = (0.5*(1+(atan(pop(i).Position))/pi)) > bc;
              pop(i).S = abs(cos(pop(i).Position)) > bc;

            % Evaluation
            [pop(i).Cost, pop(i).Out] = CostFunction(pop(i).S);

        else
            E2= exp((-b).*gamma).*cot(alpha.*gamma);
            D2=(G2.Position-pop(i).Position);

            pop(i).Velocity = E1.*(r1.*pop(i).Velocity +w1.*D2);

                        % Apply Velocity Bounds
            pop(i).Velocity=min(max(pop(i).Velocity,VelMin),VelMax);

            % Update Position
            pop(i).Position=pop(i).Position+pop(i).Velocity;

            % Velocity Reflection
            flag=(pop(i).Position<VarMin | pop(i).Position>VarMax);
            pop(i).Velocity(flag)=-pop(i).Velocity(flag);

            % Apply Position Bounds
            pop(i).Position=min(max(pop(i).Position,VarMin),VarMax);

            % Arctan transfer (Binary Coding)
            % pop(i).S = (0.5*(1+(atan(pop(i).Position))/pi)) > bc;
              pop(i).S = abs(cos(pop(i).Position)) > bc;


            % Evaluation
             [pop(i).Cost, pop(i).Out] = CostFunction(pop(i).S);

        end


        if pop(i).Cost<G2.Cost
            G2.Cost= pop(i).Cost;
            G2.Position= pop(i).Position;
        end

        if G2.Cost<G1.Cost
            G1.Cost= G2.Cost;
            G1.Position= G2.Position;
        end

        
BestSol.Cost=G1.Cost;
 
        
     end
 

     
     %%%%%%%%%%%%%   Reinforcement operator  %%%%%%%%%%%%
     % Sort Population
     pop=SortPopulation(pop);
     POPB=pop;
     POPC=pop;

     h= randi([2,3]);
     for k=(nPop-round(nPop/h)):nPop

         i1=randi([1 nPop]);
         i2=randi([1 nPop]);
         if i1==i2
             i2=randi([1 nPop]);
         end

         d1=d-It*(d/MaxIt);

         if rand<0.5

             POPC(k).Position=POPC(k).Position+d1.*E1.*(POPC(i2).Position+POPC(i1).Position)/2;
             POPC(k).Position=min(max(POPC(k).Position,VarMin),VarMax);

             POPC(k).S = abs(cos(POPC(k).Position)) > bc;

            % Evaluation
            [ POPC(k).Cost,  POPC(k).Out] = CostFunction( POPC(k).S);

         else 

             POPC(k).Position=POPC(k).Position+d1.*E2.*(POPC(i2).Position+POPC(i1).Position)/2;
             POPC(k).Position=min(max(POPC(k).Position,VarMin),VarMax);

             POPC(k).S = abs(cos(POPC(k).Position)) > bc;

            % Evaluation
            [ POPC(k).Cost,  POPC(k).Out] = CostFunction( POPC(k).S);
         end

         if POPC(k).Cost<POPB(k).Cost
             POPB(k).Cost=POPC(k).Cost;
             POPB(k).Position=POPC(k).Position;
         else

             x=POPC(k).Position;
             pCR=unifrnd(0.4,0.6);
             i3=randi([1 nPop]);
             y=POPC(i3).Position;
             z=zeros(size(x));
             j0=randi([1 numel(x)]);
             for j=1:numel(x)
                 if j==j0 || rand<=pCR
                     z(j)=y(j);
                 else
                     z(j)=x(j);

                 end
             end


             POPC(k).Position=z;
             POPC(k).Position=min(max(POPC(k).Position,VarMin),VarMax);

             pop(i).S = abs(cos(pop(i).Position)) > bc;
            
             % Evaluation
             [ POPC(k).Cost,  POPC(k).Out] = CostFunction( POPC(k).S);

             if POPC(k).Cost<POPB(k).Cost
                 POPB(k).Cost=POPC(k).Cost;
                 POPB(k).Position=POPC(k).Position;
             end

         end

         if POPB(k).Cost<G2.Cost
             G2.Cost= POPB(k).Cost;
             G2.Position= POPB(k).Position;
         end

         if G2.Cost<G1.Cost
             G1.Cost= G2.Cost;
             G1.Position= G2.Position;
         end


       %  BestSol.Cost=G1.Cost;
         BestSol=G1;

     end
     pop=POPB;

     % Store Best Cost
     BestCost(It)=BestSol.Cost;

     % disp(['Iteration ' num2str(It) ':  Best Cost = ' num2str(BestCost(It))]);

end
    

%% Results
Time = toc;
% figure;

% plot(1:MaxIt,BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% title('BTCOA Trend Feature Selection')

Out = BestSol.Out;
Out.Time = Time;
end
