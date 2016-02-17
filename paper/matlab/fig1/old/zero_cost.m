function [ cost ] = zero_cost( )

cost = struct;
cost.cost = 0;
cost.memory = 0;
cost.stack = 0;
cost.memoized = 0;

end

