function [ r ] = append_cost( a, b )

r = struct;
r.memory = max(a.memory, b.memory);
r.cost   = a.cost + b.cost;
r.stack  = max(a.stack, b.stack);
r.memoized = a.memoized + b.memoized;

end

