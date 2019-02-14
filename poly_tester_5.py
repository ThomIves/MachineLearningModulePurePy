# PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)

def poly(order=3,var=1,vars=2,powers=[0,0],memo=set()):
    # powers=[0]*vars
    for pow in range(order+1):
        powers[var-1] = pow
        if sum(powers) <= order:
            memo.add(tuple(powers))
        # print('\t', var, pow, powers)
        if var < vars:
            poly(var=var+1,powers=powers,memo=memo)

    memo = [list(x) for x in memo]
    memo.sort()
    return memo

my_ans = poly()

print(len(my_ans))
print(my_ans)