import random
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# ROBOTa FEATURE SELECTION
# -----------------------------
def fitness_function(X, Y, feature_subset, alpha=0.9, beta=0.1):
    """Evaluate subset quality using classifier performance + subset size"""
    if sum(feature_subset) == 0:
        return 0
    selected_features = [i for i, bit in enumerate(feature_subset) if bit == 1]
    X_subset = X[:, selected_features]

    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_subset, Y)
    y_pred = clf.predict(X_subset)
    score = f1_score(Y, y_pred, average='macro')

    size_penalty = len(selected_features) / X.shape[1]
    return alpha * score - beta * size_penalty

def initialize_population(pop_size, n_features):
    """Create binary-encoded population"""
    return [np.random.choice([0, 1], size=n_features).tolist() for _ in range(pop_size)]

def crossover(parent1, parent2):
    """Single-point crossover"""
    point = random.randint(1, len(parent1)-1)
    return parent1[:point] + parent2[point:]

def mutate(solution, mutation_rate=0.05):
    """Randomly flip bits"""
    return [1 - bit if random.random() < mutation_rate else bit for bit in solution]

def robota_feature_selection(X, Y, generations=20, pop_size=20):
    n_features = X.shape[1]
    population = initialize_population(pop_size, n_features)

    best_solution = None
    best_fitness = -np.inf

    for gen in range(generations):
        scores = [fitness_function(X, Y, sol) for sol in population]
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx]

        if scores[sorted_idx[0]] > best_fitness:
            best_fitness = scores[sorted_idx[0]]
            best_solution = population[0]

        # Selection and reproduction
        new_pop = population[:2]  # elitism
        while len(new_pop) < pop_size:
            p1, p2 = random.choices(population[:10], k=2)  # top 10 for selection
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop
        print(f"[GEN {gen+1}] Best fitness: {best_fitness:.4f}")

    return best_solution
