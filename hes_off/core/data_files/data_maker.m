name_file = 'TRY2.mat';

% test = load(name_file);
load(name_file);

%% Wind speed instances in year X
xx = wind{1};

%wind speeds for each hour of the year
for i = 1:8760
    xx(i) = rand*10;
end

wind{1}=xx;

% wind{1}

%% Wind speed instances in year Y
xx = wind{2};

%wind speeds for each hour of the year
for i = 1:8760
    xx(i) = rand*10;
end

wind{2}=xx;

% wind{2}

%% Wind speed instances in year Z
xx = wind{3};

%wind speeds for each hour of the year
for i = 1:8760
    xx(i) = rand*10;
end

wind{3}=xx;

% wind{3}

%%
save(name_file);

%%

% %% Wind speed instances in year X
% xx = test.wind{1}
% 
% %wind speeds for each hour of the year
% for i = 1:8760
%     xx(i) = rand*10;
% end
% 
% test.wind{1}=xx;
% 
% test.wind{1}
% 
% % pause
%
% %% Wind speed instances in year Y
% xx = test.wind{2}
% 
% %wind speeds for each hour of the year
% for i = 1:8760
%     xx(i) = rand*10;
% end
% 
% test.wind{2}=xx;
% 
% test.wind{2}
% 
% % pause
% 
% %% Wind speed instances in year Z
% xx = test.wind{3}
% 
% %wind speeds for each hour of the year
% for i = 1:8760
%     xx(i) = rand*10;
% end
% 
% test.wind{3}=xx;
% 
% test.wind{3}
% 
% % pause