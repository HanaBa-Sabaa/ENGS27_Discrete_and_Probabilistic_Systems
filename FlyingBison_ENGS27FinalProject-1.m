%% ENGS 27 Final Project
% Hana Ba-Sabaa
% Cara Ditmar
% Yefri Figueroa
% Naomi Miller
% Tyler Neath
% Esther Omene

clear
clc


%% Create node adjancy matrix
load('facebook_combined.txt')

% Create node adjacency matrix from facebook_combined.txt data
fbfriends=zeros(4038);
for i=1:88234
    fbfriends(facebook_combined(i,1)+1,facebook_combined(i,2))=1;
    fbfriends(facebook_combined(i,2),facebook_combined(i,1)+1)=1;
end

% No one is friends with themselves
for i=1:4038   
    fbfriends(i,i)=0;
end

% Delete entry with no friends (686)
nofriends = find(sum(fbfriends)==0);
fbfriends=fbfriends([1:nofriends-1,nofriends+1:end], [1:nofriends-1,nofriends+1:end]);

% Total number of people we are studying
numpeople = length(fbfriends);

% Graph adjacencies
g=graph(fbfriends);
plot(g)


%% Normalize matrix: get probabilities that a friend sees someone's message

% Number of friends each person has
num_friends=sum(fbfriends);

% Assume everyone posts once a day (so your feed has the same posts as your # of friends)
% The probability that you see one person's message is 1/(your # of friends)
prob_sees_post=1./num_friends;

% Multiply the node adjacency matrix by the probability of seeing posts (above)
% to get transition probability matrix
fbfriends_prob=fbfriends.*prob_sees_post;


%% Find number of mutual friends that each person has

% Find # of mutual friends for each pair: can start w/ sample of 200 to get
% estimate of max #s of mutual friends
mutualfriends=zeros(numpeople); 
for i=1:numpeople % run through friend i possibilities
    for j=1:numpeople % run through friend j possibilities
        if i~=j % no one has mutual friends with themselves
            for f=1:numpeople % find all the mutual friends of friend i and j
                % search across the ith row and down the jth column
                if((fbfriends(i,f)==1) && (fbfriends(f,j)==1))
                    mutualfriends(i,j)=mutualfriends(i,j)+1; % add count to new matrix
                end
            end
        end
    end
end

% Construct pmf of mutual friend counts
mutual_pmf=zeros(2,numpeople-2);
for l=0:numpeople-2 % possible # of mutual friends
    % check every combo of pairs
    for i=1:numpeople 
        for j=1:numpeople
            % create a row to represent # of mutual friends
            mutual_pmf(1,l+1)=l;
            % create row with the # of pairs with that # of mutual friends
            if mutualfriends(i,j)==l
                mutual_pmf(2,l+1)=mutual_pmf(2,l+1)+1;
            end
        end
    end
end
mutual_pmf(2,:)=mutual_pmf(2,:)/(sum(mutual_pmf(2,:))); % normalize it

% RESULT: most people have only 1 mutual friend, 10% have 4 or more mutual friends
% So, we will use 4 people as the threshold of being "close friends"


%% Determine "close" and "distant" friendships

% Load mutual friends matrix (made using the code in the previous section)
load('mutualfriends.mat');

% Delete entry with no friends (686)
mutualfriends=mutualfriends([1:nofriends-1,nofriends+1:end], [1:nofriends-1,nofriends+1:end]);

% Create a logical vector to determine how close of friends 2 people are
closeness=zeros(size(mutualfriends));  % default: 0=not friends
closeness(find(mutualfriends<4))=1;    % 1=distant friends
closeness(find(mutualfriends>=4))=2;   % 2=close friends

% Multiply closeness matrix by probability of seeing friend's post
% A close friend is twice more likely to see someone's post than a distant friend
fbfriends_prob=fbfriends_prob.*closeness;

% Normalize the rows to add to 1
sumrows=sum(fbfriends_prob');
fbfriends_prob=fbfriends_prob./sumrows';

% Check that it works (should sum to 1)
sum(fbfriends_prob');


%% SIMULATION 1
% Randomly select 10% of people to get the informational message (suggested ad)
% Once someone sees it, they have a 0.22 probability of posting about it
% on every following day

% Pick 10% randomly to receive the "informational" message at the start
start=randperm(numpeople,round(numpeople*0.1));

% Probability of each person voting
voting_prob=zeros(1,numpeople);

numsteps1 = 100;  % number of time steps (days) we are simulating
history = zeros(numsteps1,numpeople); % Matrix to hold results of who has 
                                      % seen the ad during the entire simulation
                                      % This is the same as the people who
                                      % can share the post

diffusion_time = 0;  % number of time steps for the ad to reach 90% of people
voting_at_diffusion_time = zeros(1,numpeople);  % save voting stats after reaching 90% of people

for k=1:numsteps1
    % 22% of people who have seen the post will actually share it for their friends to see
    % Simulate 22% of them posting it (based on a Bernoulli trial)
    posted_msg=zeros(2,length(start));
    posted_msg(1,:)=start;
    for i=1:length(posted_msg)
        posted_msg(2,i)=binornd(1,.22); % Logical: 1 if they share it, 0 if they don't
    end

    % Friends who posted in THIS time step
    friends_who_posted=start(find(posted_msg(2,:)));

    % Matrix of how many times each person has seen the message during THIS time step 
    did_they_see=zeros(1,numpeople); 
    for a=friends_who_posted  % loop through the people who posted
        for b=1:numpeople  % loop through all people
            prob_of_seeing_post=fbfriends_prob(a,b);  % from transition matrix
            % If someone had a chance of seeing the post, conduct a
            % binomial random trial to determine whether or not they saw
            % it, with p=prob_of_seeing_post
            if(prob_of_seeing_post~=0)
                did_they_see(b)=did_they_see(b)+binornd(1,prob_of_seeing_post);
                num(k,b)=did_they_see(b);
            end
        end
    end
    
    % Probability of someone voting increases by 0.0039 for each time they
    % saw a friend's post
    voting_prob=voting_prob+(.0039.*did_they_see);

    % Reset the starting people with the cumulative number of people who
    % have seen a post since the beginning of the simulation
    start=cat(2,start,find(did_they_see>0)); 
    start=unique(start);  % eliminate repeats
    history(k,start) = 1;
    
    % Check if at least 90% of people have seen a post
    if (length(start)>=0.9*length(fbfriends_prob)) && diffusion_time==0
        diffusion_time = k;
        voting_at_diffusion_time = voting_prob;
    end
end

% Add the base probability of voting (0.614) to each person
voting_prob=voting_prob+.614;
new_voting_prob = mean(voting_prob)  % Overall average probability of voting after num_steps

diffusion_time  % Number of steps for the post to reach 90% of people
voting_prob_at_diffusion_time = mean(voting_at_diffusion_time + 0.614)  % save voting stats after reaching 90% of people


%% Bar graph
% number of people who saw the ad vs time steps from Simulation 1 
% (with numsteps1 = 24)

lim=max(max(num));
count=zeros(k,lim);
for z=1:k
    for x=1:4037
        for p=1:lim
            if num(z,x)==p
                count(z,p)=count(z,p)+1;     
            end
        end  
    end
end 
b=bar(count,'stacked');
grid on;
title('Number of People Who Saw the Ad vs Number of Steps');
legend('1 time','2 times','3 times','4 times','Location','Best');


%% SIMULATION 2
% Randomly select 10% of people to get the informational message (suggested ad)
% Once someone sees it, they have a 0.22 probability of posting until 7 
% days have passed

current_sharers = zeros(1,numpeople);  % everyone currently able to share the post
has_seen = zeros(1,numpeople);  % a master list of everyone who has seen the ad since the beginning of the simulation

% Pick 10% randomly to receive the "informational" message at the start
start=randperm(numpeople,round(numpeople*0.1));
current_sharers(start) = 1;  % these people are on their 1st day of seeing the ad

% Probability of each person voting
voting_prob=zeros(1,4037);

numsteps2 = 23;  % number of time steps (days) we are simulating
history_sharers = zeros(numsteps2,numpeople);  % matrix to hold results of sharers during the entire simulation
history_seen = zeros(numsteps2,numpeople);  % matrix to hold results of who has seen the ad during the entire simulation
history_voting=zeros(1,numsteps2);

diffusion_time = 0;  % number of time steps for the ad to reach 90% of people
voting_at_diffusion_time = zeros(1,numpeople);  % save voting stats after reaching 90% of people

for k=1:numsteps2
    % 22% of people who have seen the post will actually share it for their friends to see
    % Simulate 22% of them posting it (based on a Bernoulli trial)
    posted_msg=zeros(2,length(start));
    posted_msg(1,:)=start;
    for i=1:length(posted_msg)
        posted_msg(2,i)=binornd(1,.22); % Logical: 1 if they share it, 0 if they don't
    end

    % Friends who posted in THIS time step
    friends_who_posted=start(find(posted_msg(2,:)));

    % Matrix of how many times each person has seen the message during THIS time step 
    did_they_see=zeros(1,numpeople); 
    for a=friends_who_posted  % loop through the people who posted
        for b=1:numpeople  % loop through all people
            prob_of_seeing_post=fbfriends_prob(a,b);  % from transition matrix
            % If someone had a chance of seeing the post, conduct a
            % binomial random trial to determine whether or not they saw
            % it, with p=prob_of_seeing_post
            if(prob_of_seeing_post~=0)
                did_they_see(b)=did_they_see(b)+binornd(1,prob_of_seeing_post);
                num(k,b)=did_they_see(b);
            end
        end
    end
    
    % Probability of someone voting increases by 0.0039 for each time they
    % saw a friend's post
    voting_prob=voting_prob+(.0039.*did_they_see);
    history_voting(k)=mean(voting_prob)+.614; %for plotting avg voting prob over time
   
    % Reset the starting people with the cumulative number of people who
    % have seen a post in the last 7 days
    start=cat(2,start,find(did_they_see>0));
    start=unique(start); %eliminate repeats
    
    % Add 1 to the number of days since last seen post
    current_sharers(start) = current_sharers(start)+1;
    history_sharers(k,:) = current_sharers;
    
    % Logical: people who have seen the post since beginning of simulation
    has_seen(start) = 1;
    history_seen(k,:) = has_seen;
    
    % If 7 days have passed since someone seeing the ad, they are no longer
    % able to share it
    current_sharers(find(current_sharers)>=7) = 0;
    % People who just saw the ad in this time step = 1 day since seeing ad
    current_sharers(find(did_they_see)) = 1;
    
    % start = people who can share in the next loop
    start = find(current_sharers);
    
    % Check if at least 90% of people have seen the ad at least once since 
    % the beginning of the simulation
    if (length(start)>=0.9*length(fbfriends_prob)) && diffusion_time==0
        diffusion_time = k;
        voting_at_diffusion_time = voting_prob;
    end
end

% Add the base probability of voting (0.614) to each person
voting_prob=voting_prob+.614;
new_voting_prob = mean(voting_prob)  % Overall average probability of voting after num_steps

diffusion_time  % Number of steps for the post to reach 90% of people
voting_prob_at_diffusion_time = mean(voting_at_diffusion_time + 0.614);  % save voting stats after reaching 90% of people


%% Bar graph
% number of people who saw the ad vs time steps from Simulation 1 
% (with numsteps2 = 23)

lim=max(max(num));
count=zeros(k,lim);
for z=1:k
    for x=1:4037
        for p=1:lim
            if num(z,x)==p
                count(z,p)=count(z,p)+1;     
            end
        end  
    end
end 
b=bar(count,'stacked');
grid on;
title('Number of People Who Saw the Ad vs Number of Steps');
legend('1 time','2 times','3 times','4 times','Location','Best');


%% Plot trends

% Sharers at Each Time Step: Simulation 1 vs Simulation 2
figure; hold on;
scatter([1:numsteps1],sum(history'~=0)/numpeople,'*')
scatter([1:numsteps2],sum(history_sharers'~=0)/numpeople,'*')
title('Sharers at Each Time Step','FontSize',16)
xlabel('Number of Days','FontSize',16)
ylabel('Percent of People in Network','FontSize',16)
l = legend('Simulation 1 (infinite memory)','Simulation 2 (7 day memory)')
l.FontSize = 16;

% Cumulative People Who Have Seen the Ad: Simulation 1 vs Simulation 2
figure; hold on;
scatter([1:numsteps1],sum(history'~=0)/numpeople,'*')
scatter([1:numsteps2],sum(history_seen'~=0)/numpeople,'*')
title('Cumulative People Who Have Seen the Ad','FontSize',16)
xlabel('Number of Days','FontSize',16)
ylabel('Percent of People in Network','FontSize',16)
l=legend('Simulation 1 (infinite memory)','Simulation 2 (7 day memory)');
l.FontSize = 16;

% Increase in voting probability: Simulation 2
figure; 
scatter([1:25],history_voting(1:25),'*')
title('Increase in Group Voting Probability', 'fontsize', 16)
xlabel('Number of Days','FontSize',16)
ylabel('Percent of People Likely to Vote','FontSize',16)


%% SIMULATION 3
% Runs the above simulation 100 times to find the average # of time steps for
% the message to reach 65% of people

average_steps_to_reach=zeros(1,100);
 for j=1:100
    current_sharers = zeros(1,4037);
    has_seen = zeros(1,4037);

    %pick 10% randomly to receive the "informational" message
    start=randperm(4037,404);

    current_sharers(start) = 1; %add second row to keep track of days since seeing ad
    %seen_post=zeros(1,4037);

    voting_prob=zeros(1,4037);

    for k=1:200

        % 22% of sharers will actually share it to their feed for friends to see
        posted_msg=zeros(2,length(start));
        posted_msg(1,:)=start;

        %simulate 22% of them posting it (bernoulli trials)
        for i=1:length(posted_msg)
            posted_msg(2,i)=binornd(1,.22);
        end

        friends_who_posted=start(find(posted_msg(2,:)));

        %matrix of how many times each person has seen the message from THIS time step 
        didtheysee=zeros(1,4037); 
        for a=friends_who_posted
            for b=1:4037 %loop through all friends
                prob_of_seeing_post=fbfriends_prob(a,b); %from transition matrix
                if(prob_of_seeing_post~=0)
                    didtheysee(b)=didtheysee(b)+binornd(1,prob_of_seeing_post);
                end
            end
        end

        voting_prob=voting_prob+(.0039.*didtheysee);

        start=cat(2,start,find(didtheysee>0));
        %the possible posters of the message in the next step (includes anyon who has seen it in the previous time step)
        start=unique(start); %eliminate repeats

        %number of days since last received post
        current_sharers(start) = current_sharers(start)+1;

        %people who have seen the post since beginning of simulation
        has_seen(start) = 1;

        %check if 7 days have passed since seeing ad
        current_sharers(find(current_sharers)>=7) = 0;

        %people who just saw it = 1 day since seeing ad
        current_sharers(find(didtheysee)) = 1;

        %people who can share it in the next loop
        start = find(current_sharers);

        % check if 65% of people have seen it in the last 7 days
        if length(find(has_seen))>=0.65*length(fbfriends_final)
            break;
        end
    end

    voting_prob=voting_prob+.614; %.614= prob of someone voting as a base case
    reached_basecase=length(find(has_seen)) 
    mean(voting_prob)
    average_steps_to_reach(j)=k;
 end
 %find avg and standard deviation of time steps to reach 65%
murand=mean(average_steps_to_reach);
stdrand=std(average_steps_to_reach);


%% SIMULATION 4
% choose most connected 10% as those who receive the message first

average_steps_pop=zeros(1,100);
for j=1:100
    current_sharers = zeros(1,4037);
    has_seen = zeros(1,4037);

    %don't pick 10% randomly
    % pick the 10% with the most connections to receive the "informational" message
    [friend_nums,index]=sort(num_friends, 'descend');
    mostpopular=index(1:404);
    start=(num_friends(mostpopular));

    current_sharers(start) = 1; %add second row to keep track of days since seeing ad
    %seen_post=zeros(1,4037);

    voting_prob=zeros(1,4037);

    for k=1:200

        % 22% of sharers will actually share it to their feed for friends to see
        posted_msg=zeros(2,length(start));
        posted_msg(1,:)=start;

        %simulate 22% of them posting it (bernoulli trials)
        for i=1:length(posted_msg)
            posted_msg(2,i)=binornd(1,.22);
        end

        friends_who_posted=start(find(posted_msg(2,:)));

        %matrix of how many times each person has seen the message from THIS time step 
        didtheysee=zeros(1,4037); 
        for a=friends_who_posted
            for b=1:4037 %loop through all friends
                prob_of_seeing_post=fbfriends_prob(a,b); %from transition matrix
                if(prob_of_seeing_post~=0)
                    didtheysee(b)=didtheysee(b)+binornd(1,prob_of_seeing_post);
                end
            end
        end

        voting_prob=voting_prob+(.0039.*didtheysee);

        start=cat(2,start,find(didtheysee>0));
        %the possible posters of the message in the next step (includes anyon who has seen it in the previous time step)
        start=unique(start); %eliminate repeats

        %number of days since last received post
        current_sharers(start) = current_sharers(start)+1;

        %people who have seen the post since beginning of simulation
        has_seen(start) = 1;

        %check if 7 days have passed since seeing ad
        current_sharers(find(current_sharers)>=7) = 0;

        %people who just saw it = 1 day since seeing ad
        current_sharers(find(didtheysee)) = 1;

        %people who can share it in the next loop
        start = find(current_sharers);

        % check if 65% of people have seen it in the last 7 days
        if length(find(has_seen))>=0.65*length(fbfriends_final)
            break;
        end
    end

    voting_prob=voting_prob+.614; %.614= prob of someone voting as a base case

    mean(voting_prob)
    reached_mostpop=length(find(has_seen));
    average_steps_pop(j)=k;
end
%find avg and std deviation of steps to reach 65% of people after 100
%simulations
mu=mean(average_steps_pop);
stddev=std(average_steps_pop);


%% SIMULATION 5
% choose least connected 10% as those who receive the message first

average_steps_least_pop=zeros(1,100);
for j=1:100
    current_sharers = zeros(1,4037);
    has_seen = zeros(1,4037);

    %don't pick 10% randomly
    % pick the 10% with the least connections to receive the "informational" message
    [friend_nums,index]=sort(num_friends, 'ascend');
    leastpopular=index(1:404);
    start=(num_friends(leastpopular));

    current_sharers(start) = 1; %add second row to keep track of days since seeing ad
    %seen_post=zeros(1,4037);

    voting_prob=zeros(1,4037);

    for k=1:200

        % 22% of sharers will actually share it to their feed for friends to see
        posted_msg=zeros(2,length(start));
        posted_msg(1,:)=start;

        %simulate 22% of them posting it (bernoulli trials)
        for i=1:length(posted_msg)
            posted_msg(2,i)=binornd(1,.22);
        end

        friends_who_posted=start(find(posted_msg(2,:)));

        %matrix of how many times each person has seen the message from THIS time step 
        didtheysee=zeros(1,4037); 
        for a=friends_who_posted
            for b=1:4037 %loop through all friends
                prob_of_seeing_post=fbfriends_final(a,b); %from transition matrix
                if(prob_of_seeing_post~=0)
                    didtheysee(b)=didtheysee(b)+binornd(1,prob_of_seeing_post);
                end
            end
        end

        voting_prob=voting_prob+(.0039.*didtheysee);

        start=cat(2,start,find(didtheysee>0));
        %the possible posters of the message in the next step (includes anyon who has seen it in the previous time step)
        start=unique(start); %eliminate repeats

        %number of days since last received post
        current_sharers(start) = current_sharers(start)+1;

        %people who have seen the post since beginning of simulation
        has_seen(start) = 1;

        %check if 7 days have passed since seeing ad
        current_sharers(find(current_sharers)>=7) = 0;

        %people who just saw it = 1 day since seeing ad
        current_sharers(find(didtheysee)) = 1;

        %people who can share it in the next loop
        start = find(current_sharers);

        % check if 90% of people have seen it in the last 7 days
        if length(find(has_seen))>=0.65*length(fbfriends_final)
            break;
        end
    end

    voting_prob=voting_prob+.614; %.614= prob of someone voting as a base case

    mean(voting_prob)
    reached_leastpop=length(find(has_seen));
    average_steps_least_pop(j)=k;
end
%find average and standard dev
muleast=mean(average_steps_least_pop);
stddevleast=std(average_steps_least_pop);


%% Video with memory
% Once you see it, you have a 0.22 probability of posting until 7 days have
% passed

v = VideoWriter('diffusionfinal','MPEG-4');
open(v);
current_sharers = zeros(1,4037);
has_seen = zeros(1,4037);

%pick 10% randomly to receive the "informational" message
start=randperm(4037,404);
highlight(h,start,'NodeColor','r')
s=[];
t=[];
for i=1:length(start)
    t=cat(2,t,neighbors(g,start(i))');
    s=cat(2,s,start(i)*ones(1,length(neighbors(g,start(i)))));
end
highlight(h,s,t,'EdgeColor','red')
annotation(figure(1),'textbox',...
[0.198142857142857 0.693333333333333 0.310428571428571 0.146666666666668],...
'String',{'Step Number 0'},...
'FitBoxToText','on');
for i=1:10
    frame=getframe(gcf);
    writeVideo(v,frame);
end
current_sharers(start) = 1; %add second row to keep track of days since seeing ad
%seen_post=zeros(1,4037);

voting_prob=zeros(1,4037);

for k=1:200

    % 22% of sharers will actually share it to their feed for friends to see
    posted_msg=zeros(2,length(start));
    posted_msg(1,:)=start;

    %simulate 22% of them posting it (bernoulli trials)
    for i=1:length(posted_msg)
        posted_msg(2,i)=binornd(1,.22);
    end

    friends_who_posted=start(find(posted_msg(2,:)));

    %matrix of how many times each person has seen the message from THIS time step 
    didtheysee=zeros(1,4037); 
    for a=friends_who_posted
        for b=1:4037 %loop through all friends
            prob_of_seeing_post=fbfriends_final(a,b); %from transition matrix
            if(prob_of_seeing_post~=0)
                didtheysee(b)=didtheysee(b)+binornd(1,prob_of_seeing_post);
            end
        end
    end

    voting_prob=voting_prob+(.0039.*didtheysee);

    start=cat(2,start,find(didtheysee>0));
    %the possible posters of the message in the next step (includes anyone who has seen it in the previous time step)
    start=unique(start); %eliminate repeats

    %number of days since last received post
    current_sharers(start) = current_sharers(start)+1;

    %people who have seen the post since beginning of simulation
    has_seen(start) = 1;

    highlighting=find(has_seen);
    highlight(h,highlighting,'NodeColor','r')
    s=[];
    t=[];
    for i=1:length(highlighting)
        t=cat(2,t,neighbors(g,highlighting(i))');
        s=cat(2,s,highlighting(i)*ones(1,length(neighbors(g,highlighting(i)))));
    end
    highlight(h,s,t,'EdgeColor','red')
    delete(findall(gcf,'type','annotation'))
    annotation(figure(1),'textbox',...
[0.198142857142857 0.693333333333333 0.310428571428571 0.146666666666668],...
'String',{['Step Number:' num2str(k)]},...
'FitBoxToText','on');
    for i=1:10
        frame=getframe(gcf);
        writeVideo(v,frame);
    end
    %check if 7 days have passed since seeing ad
    current_sharers(find(current_sharers)>=7) = 0;

    %people who just saw it = 1 day since seeing ad
    current_sharers(find(didtheysee)) = 1;

    %people who can share it in the next loop
    start = find(current_sharers);

    % check if 90% of people have seen it in the last 7 days
    if length(find(has_seen))>=0.65*length(fbfriends_final)
        break;
    end
end
close(v);
voting_prob=voting_prob+.614; %.614= prob of someone voting as a base case
reached_basecase=length(find(has_seen))
