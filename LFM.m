%LMF(Learning From Mistakes)

clc;
clear;
close all;
format compact;

data_filename='IRIS.txt';
fid = fopen(data_filename);
junk = fgetl(fid);
junk = fscanf(fid,'%s',1);

nin = fscanf(fid,'%d',1); %nin = number of inputs
junk = fscanf(fid,'%s',1);

nout = fscanf(fid,'%d',1); %nout = number of outputs
junk = fscanf(fid,'%s',1);

nrpat = fscanf(fid,'%d',1); %nrpat = number of patterns
A = fscanf(fid,'%f',[nin+nout,Inf]); %A = [I/O pairs]
fclose(fid);

x = A(1:nin,:); %x = input patterns as column vectors
d = A(nin+1:nin+nout,:); %d = desired output vectors

%Training/Valuation/Test number of patterns
tstpats = round((1/10)*nrpat);
valpats = 2*tstpats;
trpats = nrpat-valpats-tstpats;

%Parameters
ppc = [2, 5, 10]; %ppc = prototypes for each class
epochs = 15; %epoches
ka = [0.01, 0.1]; %regulates how fast learning rate 'a' tends to zero
a0 = 0.5; %initial learning rate
s = 0.6; %window size

%Result Matrix
Results = zeros(1,7);
r=0; 

%create waitbar-----------------------------------------------
waitbar1 = waitbar(0,'Please wait...'); % create a waitbar
total_iterations = 6; % total number of iterations
current_iteration = 1;
%-------------------------------------------------------------
start=tic;

%for each ppc...
for p=ppc

    %for each ka...
    for k=ka

        %matrices for storing the average cross-validation accuracy
        %for all 10 random permutations of the dataset
        tr_avg_scores = zeros(epochs,10);
        val_avg_scores = zeros(epochs,10);
        tst_avg_scores = zeros(epochs,10);
        elapseTime_avg = zeros(epochs,10);

        %for 10 random permutations of the dataset...
        for rand=1:10                
            
            %Random Permutation
            idx = randperm(nrpat); %randomised indexes of dataset entries
            x = x(:, idx); %randomised input matrix
            d = d(:, idx); %randomised output matrix
            
            %matrices of precision of each cross-validation step
            %for training/validation/test sets
            cvtr = zeros(epochs, 10);
            cvval = zeros(epochs, 10);
            cvtst = zeros(epochs, 10);

            elapseTime_cv = zeros(epochs,10); %training time        

            %CV Loops
            %for i...
            for i=1:10
               
                %Choose training/validation/test indexes for cross-validation
                idxtr = mod((i-1)*tstpats:(i-1)*tstpats+trpats-1, nrpat)+1;
                idxval = mod((i-1)*tstpats+trpats:(i-1)*tstpats+trpats+valpats-1, nrpat)+1;                    
                idxtst = mod((i-1)*tstpats+valpats+trpats:i*tstpats+trpats+valpats-1, nrpat)+1;

                %input matrices training/validation/test
                %according to above index sets
                Ptr = x(:,idxtr);    
                Pval = x(:,idxval);  
                Ptst = x(:,idxtst);  

                %output matrices training/validation/test
                %according to above index sets
                dtr = d(:,idxtr);
                dval = d(:,idxval);
                dtst = d(:,idxtst);

                %output label indexing from vector to a single value  
                trlbl = vec2ind(dtr);
                vallbl = vec2ind(dval);
                tstlbl = vec2ind(dtst);

                %Labels
                L = []; 
                for j = 1:p
                    L=[L,1:3];
                end

                %PROTOTYPE INITIALIATION                                     
                %choose p patterns from Ptr for each unique class of 
                %the dataset and add them to matrix W
                W = zeros(nout*p,nin); %weight matrix    
                nn = p*nout; %number of neurons
                z=1;
                for c=1:nn
                    while(trlbl(z)~=L(c)) %
                        z=mod(z,trpats)+1;
                    end
                    W (c,:) = Ptr(:,z); 
                end
              
                %WEIGHTS CORRECTION
                %for e epoches do...
                et=0; %elapse time
                for e = 1:epochs 

                    tic; %start timer

                    %iterate through current training patterns
                    for iter=1:trpats

                        t = (e-1)*trpats+iter-1; %iteration number
                        a = a0/(1+k*t); %learning rate for iteration t
                    
                        %Calculate Euclidean Dist from all prototypes Z
                        %for pattern 'iter' and find the 2 closest
                        %prototypes (winner, losser)
                        Z = dist(W,Ptr(:,iter));
                        Z_sorted=sort(Z);

                        %check if we have 2 prototypes with the same
                        %distance from the pattern to avoid errors in
                        %the program of variables winner/losser being a
                        %2x1 or bigger matrix instead of a 1x1
                        winner = find(Z==Z_sorted(1));
                        winner = winner(1);                             

                        %CHECK CONDITIONS FOR LEARNING OCCURANCE
                        %1.If winner belongs to the same class as the training point DO NOTHING
                        %2.If the winner belongs to a class other than
                        %that of the training point is moved away & the
                        %closest prototype of the same class is moved closer                            
                        if trlbl(iter)~=L(winner)                             
                            W(winner,:) = W(winner,:)-a*(transpose(Ptr(:,winner))-W(winner,:)); %move away
                            z=2;
                            while trlbl(iter)~=L(find(Z==Z_sorted(z)))
                                z=z+1;
                            end
                            losser=find(Z==Z_sorted(z));
                            W(losser,:) = W(losser,:)+a*(transpose(Ptr(:,losser))-W(losser,:)); %move closer                     
                        end   
                       
                        et=et+toc; %Calculate total time for all epochs until now
                        elapseTime_cv (e,i)= et; %Stores time

                    end 

                    %CALCULATE THE PERCENTAGE OF CORRECT CLASSIFICATION FOR
                    %ALL SETS (TRAINING/VALUATION/TEST) FOR THE CURRENT
                    %CROSS VALIDATION LOOP AND ADD IT TO MATRICES
                    %cvtr, cvval, cvtst

                    %for all patterns of the training set calculate
                    %distances from all prototypes and check if the winner
                    %is of the same class with the pattern
                    tr_correct_classification=0;                 
                    for h=1:trpats
                        tr_dist = dist(W,Ptr(:,h));                        
                        winner_tr = find(tr_dist==min(tr_dist));
                        if trlbl(h)==L(winner_tr)
                            tr_correct_classification = tr_correct_classification+1;
                        end
                    end
                    cvtr(e,i) =  tr_correct_classification/trpats;

                    %for all patterns of the validation set calculate
                    %distances from all prototypes and check if the winner
                    %is of the same class with the pattern
                    val_correct_classification=0;                    
                    for h=1:valpats
                        val_dist = dist(W, Pval(:, h));
                        winner_val = find(val_dist==min(val_dist));
                        if vallbl(h)==L(winner_val)
                            val_correct_classification = val_correct_classification+1;
                        end    
                    end
                    cvval(e,i) = val_correct_classification/valpats;                   
                 
                    %for all patterns of the test set calculate
                    %distances from all prototypes and check if the winner
                    %is of the same class with the pattern
                    tst_correct_classification=0;
                    for h=1:tstpats
                        tst_dist = dist(W, Ptst(:, h));
                        winner_tst = find(tst_dist==min(tst_dist));
                        if tstlbl(h)==L(winner_tst)
                            tst_correct_classification = tst_correct_classification+1;
                        end   
                    end                                  
                    cvtst(e,i) = tst_correct_classification/tstpats;                   
                end        
            end 

            %calculate the average accuracy from all loops for each set
            tr_avg_scores(:,rand) = mean(cvtr,2);
            val_avg_scores(:,rand) = mean(cvval,2);
            tst_avg_scores(:,rand) = mean(cvtst,2);
            elapseTime_avg(:,rand) = mean(elapseTime_cv,2); %average training time of all cv loops

        end      

        %calculate the average accuracy from all random permutations
        %for each set
        tr_accuracy = mean(tr_avg_scores,2);
        val_accuracy = mean(val_avg_scores,2);
        tst_accuracy = mean(tst_avg_scores,2); 
        elapseTime = mean(elapseTime_avg,2); %average training time of all permutations       
        
        %add results to matrix
        for i=1:e
            Results(r+i,:) = [p, i, k, elapseTime(i), tr_accuracy(i), val_accuracy(i), tst_accuracy(i)];               
        end
        r=r+e; %next available position in Results to store data

%         %-----------------------------------------------------------------
%         %create excel files of each model results
%         for i=1:e
%             separate_Results(i,:) = [p, i, k, elapseTime(i), tr_accuracy(i), val_accuracy(i), tst_accuracy(i)];               
%         end        
%         T=table(separate_Results(:,2), separate_Results(:,4),separate_Results(:,5), separate_Results(:,6), separate_Results(:,7),...
%           'VariableNames',{'Epoches', 'Elapse Time', 'Training Accuracy', 'Validation Accuracy', 'Test Accuracy'});
%         filename = ['LFM Results ', data_filename, '.xlsx'];
%         sheet_name = ['P=',num2str(p),', ka=',num2str(k)];
%         writetable(T, filename, "FileType","spreadsheet","Sheet",sheet_name);
%         %-----------------------------------------------------------------
        
        %waitbar update------------------------------------------------------ 
        completion = current_iteration / total_iterations; % calculate completion percentage   
        current_iteration=current_iteration+1;
        waitbar(completion,waitbar1,['Completion: ',num2str(round(completion*100)),'%']); % update waitbar            
        %--------------------------------------------------------------------------------------------------

    end
end

%Create table with all results
T=table(Results(:,1), Results(:,2), Results(:,3), Results(:,4),Results(:,5), Results(:,6), Results(:,7),...
            'VariableNames',{'Prototypes', 'Epoches', 'Ka', 'Elapse Time', 'Training Accuracy', 'Validation Accuracy', 'Test Accuracy'});

%Find model with best validation set accuracy
best_accuracy = find(Results(:,6)==max(Results(:,6))); %index of best accuracy in Results
best_accuracy=best_accuracy(1); %if we have 2 or more same accuracies we choose the first
p=Results(best_accuracy,1); %prototypes of best model
k=Results(best_accuracy,3); %ka of best model
e=Results(best_accuracy,2); %epoches for best accuracy of the model

%Create matrix with the results of the best model 
idx_best_accuracy = best_accuracy-e+1; %index of the 1st epoch of best model in result table
best_model = Results(idx_best_accuracy:idx_best_accuracy+epochs-1,:);

%Graph for all sets and epoches for best model
title_txt='Diagram of accuracies per epoch of each set.'; %Graph title
sub_txt1 = ['For best model based on validation accuracy (', num2str(max(best_model(:,6))), ').']; %Graph subtitle #1
sub_txt2 = ['Best Model Variables: prototypes = ', num2str(p),', ka = ', num2str(k), ' & epochs = ', num2str(e)]; %Graph subtitle #2
sub_txt = [{sub_txt1 }, {sub_txt2}]; %Graph subtitle

figure('Name', 'Accusracy Diagram', 'NumberTitle','off');
plot(best_model(:,2),best_model(:,5),'r', 'DisplayName', 'Training', marker='.'), hold on
plot(best_model(:,2),best_model(:,6),'g', 'DisplayName', 'Validation', marker='.') 
plot(best_model(:,2),best_model(:,7),'b', 'DisplayName', 'Test', marker='.'), grid on
My_LGD = legend; %add legend
title(title_txt); %add title
subtitle(sub_txt); %add subtitle

%---------------------------------
close(waitbar1); % close waitbar
%--------------------------------