def find_integer_combinations(n, total_sum):
    combinations = []
    stack = [(total_sum, n, [])]

    while stack:
        curr_sum, curr_n, curr_combination = stack.pop()

        if curr_n == 0:
            if curr_sum == 0:
                combinations.append(curr_combination)
            continue

        for i in range(min(total_sum+1, curr_sum + 1)):
            stack.append((curr_sum - i, curr_n - 1, curr_combination + [i]))

    return combinations

comb = find_integer_combinations(3,100)

print("comb = ",comb)
