# LOG 
### *Arturo Adelfio s316716*
#####

# LAB 1 - Set Covering problem solved with A* algorithm

*I have worked with Laura Amoroso*

```python
PROBLEM_SIZE = 30
NUM_SETS = 100
SETS = tuple(
    np.array([random() < 0.3 for _ in range(PROBLEM_SIZE)])
    for _ in range(NUM_SETS)
)
State = namedtuple('State', ['taken', 'not_taken'])
```

In order to minimize the number of taken set, we used the lenght of the taken set as the cost of A*

```python
def cost(state):
    return len(state.taken)
```

At first we tried this function for the distance in which we used the number of remained uncovered elements as a distance to the goal, but then we understood that this was wrong because it was pessimistic. We also report a first attempt of calculating the overlapping, that we then changed because it wasn't correct.

```python
#UNUSED
#kept just for showing our previous steps
def distance(state, action):

    taken_sets=list(state.taken)
    print("taken sets", taken_sets)
    #print("action",action)
    currently_covered=reduce(
        np.logical_or,
        [SETS[i] for i in taken_sets],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )
    #print("currently_covered",currently_covered)
 
    if action:
        newly_covered=SETS[action]
        #print("newly_covered", newly_covered)
        difference=0
        for i in range(len(newly_covered)):
            if (newly_covered[i] == currently_covered[i] ):
                difference+=1
        #print("difference",difference)
        overlap=difference
        taken_sets.append(action)
        

    else:
        overlap=0
        
    distance=PROBLEM_SIZE - sum(
            reduce(
                np.logical_or,
                [SETS[i] for i in taken_sets],
                np.array([False for _ in range(PROBLEM_SIZE)]),
            ))
    #print("overlap",overlap)
    print("distance",distance)
    return overlap*distance
```
Then we started from trying the solution given by the Professor to see what happened. We report it here just to make a comparison with its improved version made by us

```python
def old_h(state):
    
    already_covered = reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )
   
    if np.all(already_covered):
        return 0
    largest_set_size = max(sum(np.logical_and(s, np.logical_not(already_covered))) for s in SETS)
    missing_size = PROBLEM_SIZE - sum(already_covered)
    optimistic_estimate = ceil(missing_size / largest_set_size)
    #print("distance", optimistic_estimate, state.taken)
    return optimistic_estimate

def old_f(state):
    return old_h(state)+cost(state)
```

The idea to improve the metric is that, given the same distance (we imposed this constraint using the weights), the search continues on the node that expects the lowest overlapping in the future nodes. To compute the overlapping we used the "not" operation of the xor between the current and the future sets, to obtain the number of times that there was an overlapping (True-True or False-False)

```python
def goal_check(state):
    return np.all(reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    ))

def h(state):
    
    already_covered = reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )
   
    if np.all(already_covered):
        return 0
    largest_set_size = max(sum(np.logical_and(s, np.logical_not(already_covered))) for s in SETS)
    missing_size = PROBLEM_SIZE - sum(already_covered)
    optimistic_estimate = ceil(missing_size / largest_set_size)

    new_metric=max(sum(np.logical_not(np.logical_xor(SETS[s], already_covered))) for s in state.not_taken)
    #print("new distance", optimistic_estimate*0.9+0.1*new_metric, state.taken)
    return optimistic_estimate*0.9+0.1*new_metric
    
def f(state):
    return h(state)+cost(state)

assert goal_check(
    State(set(range(NUM_SETS)), set())
), "Problem not solvable"
```

```python
def old_astar():
    frontier = PriorityQueue()
    state = State(set(), set(range(NUM_SETS)))
    frontier.put((f(state), state))

    counter = 0
    _, current_state = frontier.get()
    #print("initial current state", current_state)
    while not goal_check(current_state):
        counter += 1
        for action in current_state[1]:
            new_state = State(
                current_state.taken ^ {action},
                current_state.not_taken ^ {action},
            )
            #print("new state", new_state)
            frontier.put((old_f(new_state), new_state))
    
        _, current_state = frontier.get()
        #print("current state",current_state)
    print("Old solution", current_state.taken)
    print(
        f"Solved in {counter:,} steps ({len(current_state.taken)} tiles)"
    )
def new_astar():
    frontier = PriorityQueue()
    state = State(set(), set(range(NUM_SETS)))
    frontier.put((f(state), state))

    counter = 0
    _, current_state = frontier.get()
    #print("initial current state", current_state)
    while not goal_check(current_state):
        counter += 1
        for action in current_state[1]:
            new_state = State(
                current_state.taken ^ {action},
                current_state.not_taken ^ {action},
            )
            #print("new state", new_state)
            frontier.put((f(new_state), new_state))
    
        _, current_state = frontier.get()
        #print("current state",current_state)
    print("New solution", current_state.taken)
    print(
        f"Solved in {counter:,} steps ({len(current_state.taken)} tiles)"
    )

    old_astar()
    new_astar()
```

Old solution {50, 99, 63}
Solved in 67 steps (3 tiles)
New solution {50, 99, 63}
Solved in 6 steps (3 tiles)

As it can be seen from the results, we tried to reach the best solution minimizing also the overlapping between the sets. 

# HALLOWEEN CHALLENGE
*I worked with Laura Amoroso*

Find the best solution with the fewest calls to the fitness functions for:

- num_points = [100, 1_000, 5_000]
- num_sets = num_points
- density = [.3, .7]


```python
num_points=100
num_sets=100
x = make_set_covering_problem(num_points, num_sets, .3)
counter=0
print("Element at row=42 and column=42:", x[42,42])
```

```python
def fitness(state):
    global counter
    counter+=1
    cost = sum(state)
    l=np.empty((num_sets,num_points),dtype=bool)
    not_taken=np.empty((num_sets,num_points),dtype=bool)


    for j in range(num_sets):
        for i in range(num_points):
            if(state[j]):
                l[j,i]=x[j,i]
            else:
                not_taken[j,i]=x[j,i]

    
    already_covered = reduce(
        np.logical_or,
        [l[i] for i in range(num_sets) if state[i]],
        np.array([False for _ in range(num_sets)]),
    )
   
    valid = np.sum(
        already_covered
    )
    
    
    #new_metric=max(sum(np.logical_not(np.logical_xor(not_taken[i],already_covered ))) for i in range(num_sets) if not state[i])
    #print(valid,-cost)
    return valid, -cost if valid else 0

used_indeces=[]
def tweak(state):
    global used_indeces
    new_state = copy(state)

    index = randint(0, num_sets - 1)

    new_state[index] = not new_state[index]
    
    return new_state
```

```python
def new_tweak(state):
    new_state = copy(state)

    taken=[]
    not_taken=[]
    covered=np.empty((num_sets,num_points),dtype=bool)
    uncovered=np.empty((num_sets,num_points),dtype=bool)

    for j in range(num_sets):
        for i in range(num_points):
            if(state[j]):
                taken.append(j)
                covered[j,i]=x[j,i]
            else:
                not_taken.append(j)
                uncovered[j,i]=x[j,i]
    

    index=choice(not_taken)

    new_state[index] = True
    for i in range(num_points):
        covered[index,i]=x[index,i]

    sum=0
    current_max=0
    current_max_index=0
    #I removed the set with the highest number of false elements
    for j in range(num_sets):
        for i in range(num_points):
            if covered[j,i]==False:
                sum+=1
        if sum>current_max:
            current_max=sum
            current_max_index=j
    new_state[current_max_index] = False

    return new_state
```

## HILL CLIMBING
```python
def hill_climbing():
    global counter
    counter=0
    current_state = [choice([False, False, False, False, False, False]) for _ in range(num_sets)]
    ended=False
    is_better=True
    not_improving=0
    while (not ended) or not_improving>100:
        new_state = new_tweak(current_state)

        new_f=fitness(new_state)
        
        new_covered_points=new_f[0]

        if new_covered_points==num_points:
            ended=True

        is_better=new_f>fitness(current_state)

        if is_better:
            not_improving=not_improving-1
            current_state = new_state
        else:
            not_improving+=1
    print( f"Solved in {counter:,} steps")
    print("final solution", fitness(current_state))

hill_climbing()
```

Solved in 24 steps
final solution (100, -10)

## ITERATED LOCAL SEARCH

```python
def hill_climbing_ils(current_state): 
    
    ended=False
    is_better=True
    not_improving=0

    while (not ended) or not_improving>100:
       
        new_state = new_tweak(current_state)

        new_f=fitness(new_state)
        
        new_covered_points=new_f[0]

        if new_covered_points==num_points:
            ended=True

        is_better=new_f>fitness(current_state)

        if is_better:
            not_improving=not_improving-1
            current_state = new_state
        else:
            not_improving=not_improving+1
    
    return current_state
def iterated_local_search():
    global counter
    counter=0
    current_state = [choice([False, False, False, False, False, False]) for _ in range(num_sets)]
    best_solution = current_state
   
    
    for i in range(5):
        print(i)
        current_solution = hill_climbing_ils(tweak(best_solution))
    
        if  fitness(current_solution) > fitness(best_solution):
            best_solution = current_solution
           
        print(fitness(best_solution))
    print( f"Solved in {counter:,} steps")
    print("final solution", fitness(best_solution))
    
iterated_local_search()
```

0
(100, -9)
1
(100, -9)
2
(100, -9)
3
(100, -9)
4
(100, -9)
Solved in 53 steps
final solution (100, -9)


```python
def iterated_local_search_perturbated():
    global counter
    counter=0
    current_state = [choice([False, False, False, False, False, False]) for _ in range(num_sets)]
    best_solution = current_state
    perturbation = current_state
    
    for i in range(5):
        print(i)
        current_solution = hill_climbing_ils(perturbation)
    
        if  fitness(current_solution) > fitness(best_solution):
            best_solution = current_solution
            random = np.random.random((num_sets,))<.1
            concatenated= [best_solution,random]
            perturbation =reduce(
                            np.logical_or, 
                           concatenated, np.array([False for _ in range(num_sets)]))
            print('random', random)
            print('best_solution', best_solution)
            print('perturbation', perturbation)
            print('random', random)
            print('len', len(random))
        print(fitness(best_solution))
    print(fitness(best_solution))

#iterated_local_search_perturbated()
```

## TABU SEARCH 

```python
def is_valid(sol, current_state):
    return np.sum(sol)>0


def tabu_search():
    global counter
    counter=0
    tabu_list=[]
    current_state = [choice([False, False, False, False, False, False]) for _ in range(num_sets)]
    
    for step in range(100):   
        print(step)
        tmp=(tweak(current_state) for _ in range(num_points))
        candidates=[(sol,fitness(sol)) for sol in tmp if is_valid(sol,current_state) and sol not in tabu_list]
        
        if not candidates:
            continue;
        else:
            max_sol= max(candidates, key=lambda x: x[1])

            if(fitness(max_sol[0])>fitness(current_state)):
                current_state=max_sol[0]
            tabu_list.append(current_state)
   
    current_state=max(tabu_list, key=lambda x:fitness(x))
    print( f"Solved in {counter:,} evaluations")
    print(fitness(current_state))
tabu_search()
```

Solved in 10,193 evaluations
(100, -6)

## SIMULATED ANNEALING

```python
def simulated_annealing():
    global counter
    counter=0
    current_state = [choice([False, False, False, False, False, False]) for _ in range(num_sets)]
   
    t=num_points
    ended=False
    is_better=True
    iteration=0
    not_improving=0

    while (not ended) or not_improving < 1:
        iteration=iteration+1
        new_state = new_tweak(current_state)

        new_f=fitness(new_state)
        
        new_covered_points=new_f[0]

        is_better=new_f>fitness(current_state)

        if new_covered_points==100:
            ended=True
       
        if not is_better:
            not_improving=not_improving+1
            sottrazione = tuple(y-x  for x, y in zip(fitness(current_state), fitness(new_state)))

            minimum=-(abs(sottrazione[0])+abs(sottrazione[1]))
            if t<=0:
                p=0
            else:    
                esponente=minimum/t            
                p=math.exp(esponente)
                
           
        if is_better or random() < p:
            if is_better:
                not_improving=not_improving-1
            current_state = new_state
       
        alpha=num_points-new_covered_points
        t=t*0.5
    print(not_improving)
    print(not ended)
    print( f"Solved in {counter:,} steps")
    print("finale state", fitness(current_state))

simulated_annealing()
```

Solved in 82 steps
finale state (100, -13)

# LAB 2 - NIM GAME
*I have worked with Laura Amoroso*

## TASK
Write agents able to play Nim, with an arbitrary number of rows and an upper bound 
on the number of objects that can be removed in a turn (a.k.a., subtraction game).

The goal of the game is to avoid taking the last object.

Task2.1: An agent using fixed rules based on nim-sum (i.e., an expert system)
Task2.2: An agent using evolved rules using ES

Nimply = namedtuple("Nimply", "row, num_objects")
sigma = 0.3

```python
class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects
def pure_random(state: Nim,weights=None) -> Nimply:
    """A completely random move"""
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)

def nim_sum(state: Nim) -> int:
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    #print(tmp)
    xor = tmp.sum(axis=0) % 2
    #print(xor)
    return int("".join(str(_) for _ in xor), base=2)

def gabriele(state: Nim,weights=None) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))

def player1(state:Nim,weights=None)->Nimply:
    """Pick always the maximum possible number of the biggest row""" 
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (m[1])))

def player2(state:Nim,weights=None)->Nimply:
    """Pick always the 1 number of the biggest row""" 
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    row=max(possible_moves, key=lambda m: (m[1]))
    return Nimply(row[0],1)

def player3(state:Nim,weights=None)->Nimply:
    """Pick always the 1 number of the lowest row""" 
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    row=min(possible_moves, key=lambda m: (m[1]))
    return Nimply(row[0],1)

def analize(raw: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = dict()
    for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):
        tmp = deepcopy(raw)
        tmp.nimming(ply)
        cooked["possible_moves"][ply] = nim_sum(tmp)
    return cooked

def optimal(state: Nim) -> Nimply: 
    analysis = analize(state) 
    logging.debug(f"analysis:\n{pformat(analysis)}") 
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0] 
    if not spicy_moves: 
        spicy_moves = list(analysis["possible_moves"].keys()) 
    ply = random.choice(spicy_moves) 
    return ply
def match(strategy0,strategy1,weights)->bool:
    #logging.getLogger().setLevel(logging.INFO)

    #strategy = (gabriele, move)
    strategy=(strategy0, strategy1)
    nim = Nim(4)
    #logging.info(f"init : {nim}")
    player = 0
    while nim:
        ply = strategy[player](nim,weights)
        #print(f"ply: player {player} plays {ply}")
        #logging.info(f"ply: player {player} plays {ply}")
        nim.nimming(ply)
        #logging.info(f"status: {nim}")
        player = 1 - player
    #logging.info(f"status: Player {player} won!")
    return player
```

Trying to play games with different strategies, we realized that the optimal strategy proposed was not the best since it sometime loses against gabriele or against the random strategy.

In the example below player 0 (optimal) is playing against player 1 (random):

INFO:root:init : <1 3 5 7 9>

INFO:root:ply: player 0 plays Nimply(row=3, num_objects=7)

INFO:root:status: <1 3 5 0 9>

INFO:root:ply: player 1 plays Nimply(row=4, num_objects=8)

INFO:root:status: <1 3 5 0 1>

INFO:root:ply: player 0 plays Nimply(row=2, num_objects=4)

INFO:root:status: <1 3 1 0 1>

INFO:root:ply: player 1 plays Nimply(row=0, num_objects=1)

INFO:root:status: <0 3 1 0 1>

INFO:root:ply: player 0 plays Nimply(row=4, num_objects=1)

INFO:root:status: <0 3 1 0 0>

INFO:root:ply: player 1 plays Nimply(row=1, num_objects=3)

INFO:root:status: <0 0 1 0 0>

INFO:root:ply: player 0 plays Nimply(row=2, num_objects=1)

INFO:root:status: <0 0 0 0 0>

INFO:root:status: Player 1 won!

Therefore we implemented the following strategy that mantains the xor = 0 since we arrive to a stopping point where the strategy becomes fixed. In fact, when we have just one row with more than one element we should change the nim sum strategy to mantain an odd number of rows with 1 element.

```python
def expert_agent(state: Nim,weights=None) -> Nimply:
    """Follow the min sum !=0 expect for the last moves""" 
    analysis = analize(state)
    
    one_stick_row=state.rows.count(1)
    more_one_stick_rows = len(state.rows)-one_stick_row-state.rows.count(0)
    

    if more_one_stick_rows==1:
        #print("if more_one")
       
        element=0
        for r in state.rows:
            if r>1:
                element=r
        row_index=state.rows.index(element)
        #ex INFO:root:status: <0 2 1 1 1>
        #if the number of rows with 1 is even leave 1 stick
        #otherwise leave 0
        if one_stick_row % 2==0:
            #print("if one stick")
            return Nimply(row_index,element-1)
        else:
            return Nimply(row_index,element)  
    

    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns == 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    #print(analysis["possible_moves"].values())
    
    
    ply = random.choice(spicy_moves)
    return ply
```

```python
num_eras=50
strategies=[pure_random, gabriele, optimal, player1, player2,player3,expert_agent]
#strategies=[pure_random, gabriele, optimal, expert_agent, player2,player3]

def evolving_strategy(state:Nim,w)->Nimply:
    moves=[]
    dic={}   
    voting={}
    for s in strategies:
        #I collect the suggested moves for each strategy
        moves.append(s(state))

    for i in range(len(moves)) :
    
        if moves[i] in dic.keys():
            dic[moves[i]].append(w[i])
        else:
            dic[moves[i]]=[w[i]]

    for key, value in dic.items():
    
        #for each move i count how many strategies suggested it 
        #and sum the relative weights
        voting[key]=sum(value)+len(value)
    
    #print("dict", dic)
    #print("voting",voting)
    #print(voting)
    max_key = max(voting, key=voting.get)
    #print(max_key)
    return max_key

def fitness(w):
    
    counter=0
    #index=w.index(max(w))
    #print(index)
    for era in range(num_eras):
        
        if(era<num_eras/2):
            if match(evolving_strategy,expert_agent,w)==0:
                counter+=1
            
            if match(evolving_strategy,pure_random,w)==0:
                counter+=1
                      
            if match(evolving_strategy,gabriele,w)==0:
                counter+=1 
           
    
        else:
            if match(expert_agent,evolving_strategy,w)==1:
                counter+=1
            if match(pure_random,evolving_strategy,w)==1:
                counter+=1
            
            if match(gabriele,evolving_strategy,w)==1:
                counter+=1 
            
    #print("games won ", counter)
    return counter
```

## (1+λ) Strategy

```python
num_iterations=100
l=5
#initialize the weights
weights=[]
for _ in range(len(strategies)):
    weights.append(random.random())

print("weights",weights)

index=weights.index(max(weights))

print("max index", index)

prev_won=0
new_won=0
improvements=0
for it in range(num_iterations):
    print("iteration n: ", it)
    counter=fitness(weights)
    
    new_counters=[]
    new_counters={}
    for _ in range(l):
        new_weights=[]
        for i in range(len(weights)):
            #tweak the weights
            new_weights.append(weights[i]+normal(0.0,sigma))
        #print(new_weights)
        #new_counters[new_weights]=fitness(new_weights)
        #print("fitness new weights")
        new_fitness=fitness(new_weights)
        if(new_fitness>counter):
            improvements+=1
            weights=new_weights
            counter=new_fitness
    
    #iterations after which check
    check_it=num_iterations/10
    if (it+1)%check_it==0:
        if improvements/check_it>1/5:
            sigma*=1.1
        else:
           sigma/=1.1
        improvements=0
    
           
    print("weights", weights)
    #print("won matches", counter)

index=weights.index(max(weights))

print("max index", index)
```
weights [-0.3229737004297923, -3.3382431974548985, -0.29059023547155416, -0.41541708439017433, 1.7766578030283755, -0.6486884187760305, 4.729506287977998]
max index 6

## Peer Review 

#### *Done*

I reviewed Triet Ngo

![Alt text](image-16.png)

I reviewed Paul Raphael

![Pau](image-17.png)


#### *Received*

![Alt text](image-18.png)

![Alt text](image-19.png)

# LAB 9 
*I worked with Laura Amoroso*

The objective of the laboratory was to write a local-search algorithm (eg. an EA) able to solve the Problem instances 1, 2, 5, and 10 on a 1000-loci genomes, using a minimum number of fitness calls.

```python
LOCI = 1000
POPULATION_SIZE = 30
OFFSPRING_SIZE = 20
TOURNAMENT_SIZE = 2
MUTATION_PROBABILITY = 0.10
GENERATIONS = 5000

is_removing=False
```

```python
@dataclass
class Individual:
   
    fitness: float
    genotype: list[int]


    def __init__(self, fitness, genotype=None):
        if genotype is None:
            #genotype = list(choices([0, 1], k=LOCI))
            genotype=[choice((0, 1)) for _ in range(LOCI)]
        object.__setattr__(self, "_genotype", genotype)
        object.__setattr__(self, "_fitness", fitness(self.genotype))


    @property
    def genotype(self):
        return self._genotype

    @property
    def fitness(self):
        return self._fitness
      
    def __eq__(self, other):
        if isinstance(other, Individual):
            return self._genotype == other._genotype
        return False

    def __hash__(self):
        # Using hash() on a tuple of hashable attributes
        return hash(tuple(self._genotype))

def select_parent(pop):
    pool = [choice(pop) for _ in range(TOURNAMENT_SIZE)]
    champion = max(pool, key=lambda i: i.fitness)
    return champion
    
def bit_flip_mutate(ind: Individual, fitness) -> Individual:
    """ bit-flip mutation """
    l=len(ind.genotype)
    
    p=MUTATION_PROBABILITY
    
    for i in range(l):
        if p>= np.random.normal():
            ind.genotype[i]=1-ind.genotype[i]
    
    return Individual(fitness, ind.genotype)

def mutate(ind: Individual,fitness) -> Individual:
    """ one gene mutation """
    genotype=list(ind.genotype)
    index = randint(0, LOCI - 1)
    genotype[index] = 1 - genotype[index]
    return Individual(fitness=fitness, genotype=list(genotype))

def one_cut_xover(ind1: Individual, ind2: Individual, fitness) -> Individual:
    index = randint(0, LOCI - 1)
    offspring1 = Individual(fitness=fitness, genotype=list(ind1.genotype[:index]) + list(ind2.genotype[index:]))
    offspring2 = Individual(fitness=fitness, genotype=list(ind1.genotype[index:]) + list(ind2.genotype[:index]))
   
    return offspring1,offspring2

def two_cut_xover(ind1: Individual, ind2: Individual,fitness) -> Individual:
    cut_point_1= randint(0, LOCI-1)
    cut_point_2= randint(0, LOCI-1)

    if cut_point_1>cut_point_2:
        cut_point_1,cut_point_2=cut_point_2,cut_point_1

    if cut_point_1 != cut_point_2:
        ind1.genotype[cut_point_1::cut_point_2],ind2.genotype[cut_point_1::cut_point_2]=ind2.genotype[cut_point_1::cut_point_2],ind1.genotype[cut_point_1::cut_point_2]

    return Individual(fitness=fitness,genotype= list(ind1.genotype)),Individual(fitness=fitness,genotype= list(ind2.genotype))

def uniform_cut_xover(ind1: Individual, ind2: Individual,fitness) -> Individual:

    l=len(ind1.genotype)
    p=1/l
   
    for i in range(l):
    
        if p >=np.random.normal():
            ind1.genotype[i],ind2.genotype[i]=ind2.genotype[i],ind1.genotype[i]


    return Individual(fitness=fitness,genotype= list(ind1.genotype)),Individual(fitness=fitness,genotype= list(ind2.genotype))

```

As shown in the code above, we used a slighlty different version of the classical one-cut crossover that gave us really promising results. I suggested its implementation to other colleagues during peer reviews.

```python
def generation(population,crossover,fitness, mutation, mut_prob):
    
    offspring = list()

    for i in range(OFFSPRING_SIZE):
        if random() < mut_prob:  # self-adapt mutation probability
            p = select_parent(population)
            o = mutation(p, fitness)
            offspring.append(o)
        else:
            # xover 
            p1 = select_parent(population)
            p2 = select_parent(population)
            
            if crossover==one_cut_xover:
                o1,o2= one_cut_xover(p1, p2, fitness)
    
            elif crossover==two_cut_xover:
                o1,o2=two_cut_xover(p1,p2,fitness)
            else:
                o1,o2=uniform_cut_xover(p1,p2,fitness)
            """ if random() < mut_prob:  # self-adapt mutation probability
                o1 = mutate(o1, fitness)
                o2 = mutate(o2, fitness)
            """
            offspring.append(o1)
            offspring.append(o2)
           
            i+=2
            

    #----------------remove previous duplicates------------
    if is_removing:
        population = list(set(population))    
    
    population.extend(offspring)
 
    population.sort(key=lambda i: i.fitness, reverse=True)
   
    population=population[:POPULATION_SIZE]

    best_fitness=population[0].fitness

    return best_fitness,population

```

## GA

```python
def ga(problem_size):
    
    global MUTATION_PROBABILITY 
    fitness_func = lab9_lib.make_problem(problem_size)

    population = [
        Individual(fitness_func)
        for _ in range(POPULATION_SIZE)
    ]

    best_fitnesses=[]
    num_iterations=50000
    it=0
    new_fitness=0
    increasing=0
    decreasing=0

    for i in range(num_iterations):
        last_fitness=new_fitness
        it=i+1
        new_fitness, population= generation(population, one_cut_xover, fitness_func, mutate,0.15)
        best_fitnesses.append(new_fitness)
        if i%10==0:
           print("iteration",i, "fitness",new_fitness,"calls",fitness_func.calls)
        if new_fitness > last_fitness:
            increasing+=1
            if increasing>10 and MUTATION_PROBABILITY>0.01:
                increasing=0
                MUTATION_PROBABILITY*=0.8
        else:
            decreasing+=1
            if decreasing>10 and MUTATION_PROBABILITY<0.3:
                decreasing=0
                MUTATION_PROBABILITY*=1.05
                
        
        if new_fitness==1.0:
            break;

    plt.plot( range(0,it), best_fitnesses)
    plt.title("Problem size "+ str(problem_size) + " Genetic Algorithm, fitness calls: "+ str(fitness_func.calls))
    plt.xlabel("iterations")
    plt.ylabel("fitness") 
    plt.show()
    print("calls",fitness_func.calls) 
```

![Alt text](image.png)
![Alt text](image-1.png)

```python
LOCI = 1000
POPULATION_SIZE = 50
OFFSPRING_SIZE = 40
TOURNAMENT_SIZE = 3
MUTATION_PROBABILITY = 0.10

is_removing=True
```

![Alt text](image-2.png)

```python
LOCI = 1000
POPULATION_SIZE = 50
OFFSPRING_SIZE = 30
TOURNAMENT_SIZE = 2
MUTATION_PROBABILITY = 0.05

is_removing=False
```

![Alt text](image-3.png)
![Alt text](image-4.png)

## ISLAND MODEL

```python
ALL_POPULATION_SIZE = 300
TOURNAMENT_SIZE = 2
MUTATION_PROBABILITY = 0.01
MIGRATION_PROBABILITY=0.01
LOCI=1000
num_islands=5
POPULATION_SIZE=ALL_POPULATION_SIZE//num_islands
OFFSPRING_SIZE =POPULATION_SIZE//3
is_removing=True
print(POPULATION_SIZE) 
print(OFFSPRING_SIZE)
```

60

20


```python
fitness_function=lab9_lib.make_problem(10)   

@dataclass
class Island:
    population: list[Individual]
    crossover: callable
    scale_factor: float


def migration(islands):
    #print(len(islands[0].population))
    migrants=[]
    num_migrants=10
    for i in range(len(islands)):
        for _ in range(num_migrants):
         
            if random()<0.5:
                migrant=max(islands[i].population, key=lambda x: x.fitness)
            else:
                migrant=choice(islands[i].population)
      
            islands[i].population.remove(migrant)
     
            migrants.append(migrant)

    for i in range (len(islands)):
        for j in range(len(migrants)):
            if i==0:
                islands[i].population.append(migrants[len(migrants)-j-1])
            else:
                islands[i].population.append(migrants[i*num_migrants-j-1])
                
    return islands

population = [
    Individual(fitness_function)
    for _ in range(ALL_POPULATION_SIZE)
]

islands=[]
#-----------------SIMILARITY COMPUTED VIA FITNESS ------------

"""
# Create the islands with similar individual
population.sort(key=lambda i: i.fitness)
   
j=0
for i in range(0, len(population), POPULATION_SIZE):
    j+=1
    sublist = population[i:i + POPULATION_SIZE]
    prob= random()
    if prob<1.0:
        islands.append(Island(population=sublist, crossover=one_cut_xover, scale_factor=1.0))
   
    else:
        islands.append(Island(population=sublist, crossover=one_cut_xover, scale_factor=1.5))


"""
#------------SIMILARITY COMPUTED VIA CLUSTERING--------------------

# Convert the list to a NumPy array
data_array = np.array([individual.genotype for individual in population])

# Specify the number of clusters (you can adjust this based on your needs)
num_clusters = num_islands

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data_array)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Group binary vectors by cluster
clustered_data = {i: [] for i in range(num_clusters)}
for i, vector in enumerate(population):
    cluster_label = cluster_labels[i]
    clustered_data[cluster_label].append(vector)

# Print the result
for cluster_label, individuals in clustered_data.items():
    #print(f"Cluster {cluster_label + 1}: {[ind for ind in individuals]}")
    islands.append(Island(population=individuals, crossover=one_cut_xover, scale_factor=1.0))
    #print("len islands", len(islands))

best_fitnesses=[0 for _ in range(num_islands)]
max_fitness=0
max_fitnesses=[]
calls=[]
it=0
same=0
last_fitness=0
increasing=0
decreasing=0

#while max_fitness!=1.0:
for i in range(30000):   
    
    last_fitness=max_fitness

    for j in range(len(islands)):
        
        best_fitnesses[j],islands[j].population = generation(islands[j].population,islands[j].crossover, fitness_function,mutate, islands[j].scale_factor*MUTATION_PROBABILITY)
        
    max_fitness=max(best_fitnesses)
    max_fitnesses.append(max_fitness)
    
    if i%10==0:
        print("iteration", i, "best ",  max_fitness)
        print("MUTATION",MUTATION_PROBABILITY)
        print("calls",fitness_function.calls)
        #print("MIGRATION", MIGRATION_PROBABILITY)

    if random()<MIGRATION_PROBABILITY:
        islands=migration(islands)
       
   
    if max_fitness > last_fitness:
        increasing+=1
        if increasing>30 and MUTATION_PROBABILITY>0.001:
            increasing=0
            MUTATION_PROBABILITY*=0.8
            MIGRATION_PROBABILITY*=0.8
    else:
        decreasing+=1
        if decreasing>20 and MUTATION_PROBABILITY<0.05:
            decreasing=0
            MUTATION_PROBABILITY*=1.05
            MIGRATION_PROBABILITY*=1.05

    if MUTATION_PROBABILITY>0.05:
        MUTATION_PROBABILITY*=0.75
        MIGRATION_PROBABILITY*=0.75
    
    it=i+1
    calls.append(fitness_function.calls)
    if max_fitness==1.0:
        break;
    #print(len(islands[0].population))
```

```python
max_value=max(max_fitnesses)
ind=max_fitnesses.index(max_value)
print(ind)
#plt.plot( calls[0:ind], max_fitnesses[0:ind])
plt.plot( calls, max_fitnesses)
plt.title("Problem 5 Island model, fitness calls: "+ str(calls[ind]) +"\n fitness: "+str(max_value))
plt.xlabel("calls")
plt.ylabel("fitness") 
print("calls",calls[ind]) 
```

4130
calls 810204

```python
![Alt text](image-7.png)
```

Clustering

![Alt text](image-8.png)

## HILL CLIMBING ( STEEPEST ASCENT )

```python
LOCI=1000
fitness_hc=lab9_lib.make_problem(10)

fitnesses=[]
calls=[]

def new_mutate(state:Individual, fitness)->Individual:
    """" performes the 1 bit mutation for all the bits and return all the possible individuals"""
    individuals=[]
    
    for j in range (LOCI):
        gen=copy(state.genotype)
        gen[j]= 1-gen[j]
        individuals.append(Individual(fitness, genotype=gen))
        #print(individuals[j])
    return individuals

def hill_climbing():

    counter=0
    
    ended=False
    is_better=True
    not_improving=0
    current_state=Individual(fitness_hc)
    #new_state=current_state
    is_bitflip=False

    for _ in range(500000):

        counter+=1

        if not_improving>50:
            not_improving=0
            #chooses the best according to fitness
            new_state = max(new_mutate(current_state,fitness_hc), key=lambda x:x.fitness)

        elif is_bitflip:

            new_state = bit_flip_mutate(current_state,fitness_hc)
        
        else:
            new_state= mutate(current_state, fitness_hc) 
        
        
        if current_state.fitness==1.0:
            ended=True
            break;

        is_better=new_state.fitness>current_state.fitness

        if is_better:
            not_improving=0
            current_state = new_state
            is_bitflip=False
        else:
            not_improving+=1
            is_bitflip=True

        calls.append(fitness_hc.calls)
        fitnesses.append(current_state.fitness)

        if counter%50==0:

            print("current ", current_state.fitness)
        
        #print("new", new_state.fitness)
        #print("calls", fitness_hc.calls)

    print( f"Solved in {counter:,} steps")
    print("final solution", current_state.fitness)


hill_climbing()
print(fitness_hc.calls)
max_value=max(fitnesses)
ind=fitnesses.index(max_value)
#plt.plot(calls[0:ind+1], fitnesses[0:ind+1])
plt.plot(calls, fitnesses)
plt.title("Problem 10 hill climbing,calls: "+ str(calls[ind]) +"\n fitness: "+str(max_value))
plt.xlabel("calls")
plt.ylabel("fitness") 
print("calls",calls[ind]) 
```
![Alt text](image-9.png)

## Peer Review 

#### *Done*

I reviewed Miriam Ivaldi

![Alt text](image-13.png)

```python
offspring_one = Individual(fitness=None,
                           genotype= ind1.genotype[:cut_point_one] + ind2.genotype[cut_point_one:cut_point_two] + ind1.genotype[cut_point_two:])
#before
#offspring_two = Individual(fitness=None,
#                         genotype=ind2.genotype[:cut_point] + ind2.genotype[cut_point:])
offspring_two = Individual(fitness=None,
                         genotype= ind2.genotype[:cut_point_one] + ind1.genotype[cut_point_one:cut_point_two] + ind2.genotype[cut_point_two:])
```

![Alt text](image-14.png)

I reviewed Giorgio Cacopardi

![Alt text](image-15.png)

#### *Received*

No review received. Since the lab was done with Laura Amoroso, I take the suggestion of the issuesshe received from other colleagues.

# LAB 10
I worked alone. The code was not entirely completed within the submission date due to time constraints.

```python
State = namedtuple('State', ['x', 'o'])
MAGIC = [2, 7, 6, 9, 5, 1, 4, 3, 8]
def print_board(pos):
    """Nicely prints the board"""
    for r in range(3):
        for c in range(3):
            i = r * 3 + c
            if MAGIC[i] in pos.x:
                print('X', end='')
            elif MAGIC[i] in pos.o:
                print('O', end='')
            else:
                print('.', end='')
        print()
    print()
def win(elements):
    """Checks is elements is winning"""
    return any(sum(c) == 15 for c in combinations(elements, 3))

def state_value(pos: State):
    """Evaluate state: +1 first player wins"""
    if win(pos.x):
        return 1 #first player win
    elif win(pos.o):
        return -1 #second player win
    else:
        return 0 #drawn
class RandomPlayer:
    def __init__(self) -> None:
        pass
        
    def move(self,possible_moves, state=None):
        return choice(list(possible_moves))
import random

class QAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0, exploration_decay=0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        # Initialize Q-table 
        self.q_table = {}
        self.training = True

    def move(self, possible_moves, state):
        # Exploration-exploitation trade-off
        #print(self.q_table)
        if np.random.rand() < self.exploration_prob: #Explore
            return choice(list(possible_moves))
        else: #Exploit
            #Choosing from the QTable
            action = np.argmax(self.q_table[state, :]) 
        
        
            
        return action
        

    def update_q_table(self, state, action, reward, next_state):
        # Convert next_state to frozenset if it's a set
        print("Next state", next_state)
        next_state = (frozenset(next_state.x), frozenset(next_state.o))
        state = (frozenset(state.x), frozenset(state.o))

        # Q-value update using the Bellman equation
        print(self.q_table)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros((10,))
        if state not in self.q_table:
            self.q_table[state] = np.zeros((10,))
        
        current_q_value = self.q_table[state][action]
        max_future_q_value = np.max(self.q_table[next_state])
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                    self.learning_rate * (reward + self.discount_factor * max_future_q_value)
        self.q_table[state][action] = new_q_value

        
def game(p1,p2, train = True):
    trajectory = list()
    state = State(set(), set())
    available = set(range(1, 9+1))
    
    
    game_hist = list()
    players = []
    moves=list()
    players.append(p1)
    players.append(p2)
    index = 0
    i = 0
    current_player = players[index]
    agent = None
    
    while True:
        #print(current_player, index)
        
        if index == 0:
            move = current_player.move(available, state)
           
            #print(available)
            #print(move)
            state.x.add(move)
            
            moves.append(move)
            trajectory.append(deepcopy(state))
            available.remove(move)
            
            if win(state.x) or not available:
                break
        else:
            move = current_player.move(available, state)
            state.o.add(move)
            moves.append(move)
            trajectory.append(deepcopy(state))
            available.remove(move)
            #print(available)
            #print(move)
            if win(state.o) or not available:
                break
        
        
        isAgent = isinstance(current_player, QAgent)
        
        if isAgent and agent == None:
            agent = current_player
            previous_action = move
        
        if not isAgent  and i > 2: #QAgent opponent is playing, but it's not the first move, so QAgent already played
            #print(trajectory)
            previous_state = trajectory[-2]
            #print("previous state",previous_state)
            #print("reward", reward)
            #print("last_move", moves[-2])
            reward = moves[-2] // 3
                        
            agent.update_q_table(previous_state, previous_action, reward, state)
        
        i = i+1
            
        
            
        index = 1 - index
        current_player = players[index]
        result = state_value(state)
        #print("result", result) #1 first winning, -1 second winning, 0 drawn
        
    return trajectory, moves, result, available
```

```python
def train():
    player1 = RandomPlayer()
    player0 = QAgent()
    qAgent_wins = 0
    qAgent_lost = 0
    games_number = 100
    
    for _ in tqdm(range(games_number)):
        trajectory, moves, reward, possible_moves = game(player0, player1, train=True)
        
        if reward == 1: #first player (QAgent wins)
            qAgent_wins += 1
            reward = reward + 10
            state = trajectory[-1]
            move = moves[-1]
            #new_state = deepcopy(state)
            #new_state.x.add(choice(possible_moves))
        elif reward == -1:
            reward == reward -20
            qAgent_lost +=1
            state = trajectory[-2]
            move = moves[-2]
            new_state = trajectory[-1]
        else:
            state = trajectory[-1]
            move = moves[-1]
            #new_state = deepcopy(state)
            #new_state.x.add(choice(possible_moves))

        new_state = trajectory[-1]
         
        player0.update_q_table(state, move, reward, new_state)
        
 
    print("Winning/Draw", (games_number - qAgent_lost)/games_number)
    print("Lost games", qAgent_lost/games_number)   
```

## Peer Review 

#### *Done*
 No reviews, since I haven't finished the code I thought I wasn't in the position to give advices.

#### *Received*

I received a lot of reviews and suggestions. The advices were really usefull and since I used q learning in the final project, I tried to implement the suggestions given bu the colleagues.

![Alt text](image-10.png)

![Alt text](image-11.png)

![Alt text](image-12.png)

# QUIXO
*I have collaborated with Laura Amoroso (s313813)*

We tried to implement different strategies to compete against the random player and win more than 50% of the games.

## Minmax with alpha beta pruning

The following code explores the Minimax algorithm, augmented with alpha-beta pruning. By combining these techniques, we aim to create an agent capable of strategic decision-making in this complex board game.

The ***MinmaxPlayer*** plays according to the best move selected by the minimax function. Given the possible moves and the next states of the game, the algorithm generates a tree of the given depth trying to maximize the evaluation of the move to be chosen.

Since Quixo moves and states are a really large number we implemented ***alpha-beta pruning*** to reduce the number of nodes evaluated in the game tree. This helps to discard branches of the tree that are unlikely to lead to a better solution than the one already found, thus improving the algorithm's efficiency by avoiding unnecessary computations.

```python
class MinmaxPlayer(Player):
    """
    player that plays according to the minmax algorithm
    """
    def __init__(self, depth, symbol):
        self.depth = depth
        self.symbol=symbol
        self.maximizer=True

    def make_move(self,game,state=None):
        _, best_move = self.minimax(game, self.depth, True, float('-inf'), float('inf'))
        return best_move

    def minimax(self, game, depth, maximizing_player, alpha, beta):
        """
        minmax algorithm
        """
        self.maximizer=maximizing_player    
            
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game), None

        legal_moves = self.get_legal_moves(game)
        next_states = self.calculate_next_states(game, legal_moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for i, new_game in enumerate(next_states):
               
                eval, _ = self.minimax(new_game, depth - 1, False, alpha, beta)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = legal_moves[i]
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for i, new_game in enumerate(next_states):

                eval, _ = self.minimax(new_game, depth - 1, True, alpha, beta)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = legal_moves[i]                
                    beta = min(beta, min_eval)
                if beta <= alpha:
                    break
  
            return min_eval, best_move
```

*calculate_next_states* function, given the current state and the next legal moves, it generates all the next possible states that will be included in the game tree.

```python
    def calculate_next_states(self, game, legal_moves):
        next_states = []

        for move in legal_moves:
            position, direction = move
            new_game = deepcopy(game)
            new_game=self.apply_move(new_game, position, direction)
            next_states.append(new_game)

        return next_states
```

The following function creates a new board to simulate the play of a move. It basically reproduce the slide function in order to generate the board with the new move applied.

```python
    def apply_move(self, game, pos, direction):
        
        board=deepcopy(game.get_board())

        position=(pos[1],pos[0])

        if self.maximizer is True:
            board[position] = self.symbol 
        else:
            board[position] =(1-self.symbol)

        piece = board[position]
        # if the player wants to slide it to the left
        if direction == Move.LEFT:
            # for each column starting from the column of the piece and moving to the left
            for i in range(position[1], 0, -1):
                # copy the value contained in the same row and the previous column
                board[(position[0], i)] = board[(
                    position[0], i - 1)]
            # move the piece to the left
            board[(position[0], 0)] = piece
        # if the player wants to slide it to the right
        elif direction == Move.RIGHT:
            # for each column starting from the column of the piece and moving to the right
            for i in range(position[1], board.shape[1] - 1, 1):
                # copy the value contained in the same row and the following column
                board[(position[0], i)] = board[(
                    position[0], i + 1)]
            # move the piece to the right
            board[(position[0], board.shape[1] - 1)] = piece
        # if the player wants to slide it upward
        elif direction == Move.TOP:
            # for each row starting from the row of the piece and going upward
            for i in range(position[0], 0, -1):
                # copy the value contained in the same column and the previous row
                board[(i, position[1])] = board[(i - 1, position[1])]
            # move the piece up
            board[(0, position[1])] = piece

        # if the player wants to slide it downward
        elif direction == Move.BOTTOM:
            # for each row starting from the row of the piece and going downward
            for i in range(position[0], board.shape[0] - 1, 1):
                # copy the value contained in the same column and the following row
                board[(i, position[1])] = board[(
                    i + 1, position[1])]
            # move the piece down
            board[(board.shape[0] - 1, position[1])] = piece

        game.set_board(board)
        return game

```

*evaluate* function is used by the minmax algorithm to evaluate the leaves of the game tree. Here we introduced the logic for evaluating each state.
States are evaluated as it follows:
- Winning states: +10
- Losing states: -10
- Intermediate states: number of minmax player’s position subtracted by the opponent’s ones

The evaluation of the intermediate state has been implemented to avoid reaching always the leafs of the three, reducing the complexity and improving the pruning. This metric induce the player to maximize the number of his position.

```python
    def evaluate(self, game:'Game'):
        """
            function that evaluates the leaves
        """

        winner = game.check_winner()

        if winner == self.symbol:
            return 10
           
        elif winner == 1-self.symbol:
            return -10

        else:
            # return the difference between the 0 and the ones 
            count_zeros=0
            count_ones=0
            for x in range(game.get_board().shape[0]):  
                for y in range(game.get_board().shape[1]):
                    
                    el=game.get_board()[x][y]
                    #print(el)                
                    if el==0:
                        count_zeros+=1
                    elif el==1:
                        count_ones+=1

            if self.symbol==0:
                return count_zeros - count_ones
            else:
                return count_ones - count_zeros

```
The following functions calculate all the possible moves playable given the board and then checks if the move can be played by the minmax player.

```python
    def get_legal_moves(self, game:Game):
        """
        Return the legal moves
        """
        legal_moves = []

        rows, cols = game.get_board().shape
        # Top border indices
        top_indices = [(0, i) for i in range(cols)]

        # Bottom border indices
        bottom_indices = [(rows - 1, i) for i in range(cols)]

        # Left border indices (excluding corners)
        left_indices = [(i, 0) for i in range(1, rows - 1)]

        # Right border indices (excluding corners)
        right_indices = [(i, cols - 1) for i in range(1, rows - 1)]

        indices=top_indices+ bottom_indices+left_indices+right_indices

        for x,y in indices:
            for direction in Move:
                if self.is_move_playable(game, (y,x), direction):
                    legal_moves.append(((x, y), direction))
   
        return legal_moves

    def is_move_playable(self, game, position, direction):
        x,y= position

        acceptable: bool = (
            # check if it is in the first row
            (x == 0 and y < 5)
            # check if it is in the last row
            or (x == 4 and y< 5)
            # check if it is in the first column
            or (x <5 and y ==0)
            # check if it is in the last column
            or (x <5 and y == 4)
            # and check if the piece can be moved by the current player
        ) 
        if acceptable is False:
            return False
        # Check if the move is within the bounds of the board
        if not (0 <= x < game.get_board().shape[0] and 0 <= y < game.get_board().shape[1]):
            return False

        if self.maximizer:
            if game.get_board()[x, y] == 1-self.symbol:
                return False
        else:
            if game.get_board()[x, y] == self.symbol:
                return False
        # Check if the move is towards an empty cell
        if direction == Move.TOP and x==0:
            return False
        elif direction == Move.BOTTOM and x==4:
            return False
        elif direction == Move.LEFT and y == 0:
            return False
        elif direction == Move.RIGHT and y == 4:
            return False

        return True
```

## Montecarlo Q Learning
We implemented a player using a different strategy. The Montecarlo agent is trained choosing action randomly with a certain probability or choosing the best actions already evaulated in the q-table. 
Each action is evaluated and rewarded based on his effectivness and then backpropagated.

The player in initialized with a certain exploration_rate that sets the probability of choosing an action randomly. This probability is reduced gradually during the training process.

Moreover, since Quixo has a large amount of possible moves to be chosen and evaluated, we implemented some utility functions that take advantage of the board symmetries to reduce the size of the table. Given the moves, we look in the table for rotated or mirrored moves. If the current board is rotated we recompute the action according to it.

Example of equivalent states with respect to symmetries:

![Alt text](image-5.png)

This optimization reduce significantly the number of states.

Below the code used to manage this optimization.

```python
def rotate_board_90_clockwise(matrix):
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("Input matrix must be a 5x5 matrix")
 
    # Transpose the matrix
    transposed_matrix = [list(row) for row in zip(*matrix)]
 
    # Reverse each row in the transposed matrix to get the 90-degree counterclockwise rotation
    rotated_matrix = [row[::-1] for row in transposed_matrix]
 
    return np.array(rotated_matrix)

def rotate_board_90_anticlockwise(matrix):
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("Input matrix must be a 5x5 matrix")
 
    rotated_matrix = [row[::-1] for row in matrix]
    
    transposed_matrix = [list(row) for row in zip(*rotated_matrix)]
    
    return np.array(transposed_matrix)


def rotate_90_clockwise(points):
    if any(len(point) != 2 for point in points):
        raise ValueError("Each point must be a tuple of length 2")

    # Convert tuples to matrix for rotation
    matrix = [[0] * 5 for _ in range(5)]
    for x, y in points:
        matrix[x][y] = 1

    # Transpose the matrix
    transposed_matrix = [list(row) for row in zip(*matrix)]

    rotated_matrix = [row[::-1] for row in transposed_matrix]

    # Extract the rotated points from the rotated matrix
    rotated_points = [(i, j) for i, row in enumerate(rotated_matrix) for j, value in enumerate(row) if value == 1]

    return rotated_points

def rotate_90_anticlockwise(points):
    if any(len(point) != 2 for point in points):
        raise ValueError("Each point must be a tuple of length 2")

    # Convert tuples to matrix for rotation
    matrix = [[0] * 5 for _ in range(5)]
    for x, y in points:
        matrix[x][y] = 1

    rotated_matrix = [row[::-1] for row in matrix]
    
    transposed_matrix = [list(row) for row in zip(*rotated_matrix)]

    # Extract the rotated points from the rotated matrix
    rotated_points = [(i, j) for i, row in enumerate(transposed_matrix) for j, value in enumerate(row) if value == 1]

    return rotated_points
    
    
def mirror_points(points):
    if any(len(point) != 2 for point in points):
        raise ValueError("Each point must be a tuple of length 2")

    mirrored_points = [(x, 4 - y) for x, y in points]

    return mirrored_points

def mirror_board(matrix):
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Input matrix must be rectangular")
 
    mirrored_matrix = [row[::-1] for row in matrix]
 
    return np.array(mirrored_matrix)
```

### Montecarlo Agent

Below the init function and two utility function used for the transformation and rotation of the states described before.

```python
class MontecarloAgent(Player):
    """
    agent that is trained with the montecarlo algorithm of reinforcement learning
    """
    def __init__(self, symbol):
        self.q_table = {}
        self.symbol=symbol
        self._winning_games=0
        self._drawn_games=0
        self.exploration_rate=1
        self.rewards=[]
        self.gamma=0.9
        self.changing_symbol=False
        self.is_train=True



    def transform_state(self,state, board=np.empty((0,))):

        state_key = (frozenset(state[0]), frozenset(state[1]))
        original_board=board
        
        is_in_qtable = False
        rotation_type = ""


        if state_key in self.q_table:
            is_in_qtable = True
        else:
            state_key = (frozenset(set(rotate_90_clockwise(frozenset(state[0])))), frozenset(set(rotate_90_clockwise(frozenset(state[1])))))

            if state_key in self.q_table:
                if board.size!=0:
                    board = rotate_board_90_clockwise(board)
                is_in_qtable = True
                rotation_type = "clockwise"
            else:
                state_key = (frozenset(set(rotate_90_anticlockwise(frozenset(state[0])))), frozenset(set(rotate_90_anticlockwise(frozenset(state[1])))))
                if state_key in self.q_table:
                    if board.size!=0:
                        board = rotate_board_90_anticlockwise(board)
                    is_in_qtable = True
                    rotation_type = "anticlockwise"
                else:
                    state_key = (frozenset(set(mirror_points(frozenset(state[0])))), frozenset(set(mirror_points(frozenset(state[1])))))
                    if state_key in self.q_table:
                        #print("board to mirror, ", board)
                        if board.size!=0:
                            board = mirror_board(board)
                        #print("mirrored board",board)
                        is_in_qtable = True
                        rotation_type = "mirrored" 
                    else:
                        return (frozenset(state[0]), frozenset(state[1])),original_board,False, ""
        return state_key,board,is_in_qtable,rotation_type

    def transform_action(self,rotation_type,action)-> tuple[tuple[int, int], Move]:
        
        """ rotate the action according to opposite of the rotation_type"""
        #decode the action calculating the effective point (x,y) and the correct direction
        if rotation_type == "" :#board hasn't been rotated
            return action #return the action as it was calculated
        elif rotation_type == "clockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_anticlockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.BOTTOM
            elif direction == Move.BOTTOM:
                direction = Move.RIGHT
            else: 
                direction = Move.TOP
                
            #print("iniitial", action)
            #print("new", (new_point, direction) )
            return ((new_point[1], new_point[0]), direction)
        
        elif rotation_type == "anticlockwise":
            #print("anticlockwise")
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_clockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.RIGHT
            elif direction == Move.LEFT:
                direction = Move.TOP
            elif direction == Move.BOTTOM:
                direction = Move.LEFT
            else: 
                direction = Move.BOTTOM

            return ((new_point[1], new_point[0]), direction)
        else: #mirrored
            
            initial_point = action[0]
            direction = action[1]
            
            new_point = mirror_points([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.RIGHT:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.RIGHT
            return ((new_point[1], new_point[0]), direction)
        
    def rotate_action(self, rotation_type, action) -> tuple[tuple[int, int], Move]:

        """ rotate the action according to the rotation_type"""
           #decode the action calculating the effective point (x,y) and the correct direction
        if rotation_type == "" :#board hasn't been rotated
            return action #return the action as it was calculated
        elif rotation_type == "anticlockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_anticlockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.BOTTOM
            elif direction == Move.BOTTOM:
                direction = Move.RIGHT
            else: 
                direction = Move.TOP
            
            return ((new_point[1], new_point[0]), direction)
        
        elif rotation_type == "clockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_clockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.RIGHT
            elif direction == Move.LEFT:
                direction = Move.TOP
            elif direction == Move.BOTTOM:
                direction = Move.LEFT
            else: 
                direction = Move.BOTTOM

            return ((new_point[1], new_point[0]), direction)
        else: #mirrored
            
            initial_point = action[0]
            direction = action[1]
            
            new_point = mirror_points([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.RIGHT:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.RIGHT
            return ((new_point[1], new_point[0]), direction)
```

The following funciton is the one used to select the move that has to be played. During training the player chooses randomly or selecting the most rewarded action for the current state from the table.
If the action is not present in the table it is inserted. Once the action has been chosen, it is evaluated and rewarded.

```python
  def make_move(self,game, state)-> tuple[tuple[int, int], Move]:
        
        """ function that returns a move for the montecarlo agent """
        if self.gamma<0.99:
            self.gamma*=1.01

        action=None
        
        #print(state)
        board = game.get_board()
        state_key,board,is_in_qtable,rotation_type=self.transform_state(state, game.get_board())

        available_moves=list(self.get_legal_moves(board))
  
        if random() < self.exploration_rate and self.is_train:
            # sometimes make random moves
            action = choice(available_moves)
            
            if not is_in_qtable:
                self.q_table[state_key] = dict.fromkeys([action], 0)
                
            """            else:
                self.q_table[state_key][action] = 0 """
            
        else:
         
            if not is_in_qtable and self.is_train:
                self.q_table[state_key] = dict.fromkeys(available_moves, 0)
                is_in_qtable=True
            #choose the action based on the q table
            if is_in_qtable: 

               
                action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                
            
                #If the best action has a negative value, all the possible moves are added 
                if self.q_table[state_key][action] < 0 and len(self.q_table[state_key])==1:
 
                    for move in available_moves:
                        if move not in self.q_table[state_key].keys():
 
                            self.q_table[state_key][move]=0

                    action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                     
                         
            if action is None or action not in available_moves:

                action=choice(list(available_moves))

                if self.is_train:
                    self.q_table[state_key] = dict.fromkeys([action], 0) 


        if self.is_train:            
            count_zeros=0
            count_ones=0
            for x in range(game.get_board().shape[0]):  
                for y in range(game.get_board().shape[1]):
                    
                    el=game.get_board()[x][y]
                             
                    if el==0:
                        count_zeros+=1
                    elif el==1:
                        count_ones+=1
           
            if self.symbol==0:
                self.rewards.append(count_zeros - count_ones)
            else:
                self.rewards.append(count_ones - count_zeros)
        
        new_action= self.transform_action(rotation_type,action)
        return new_action
```

```python
  def add_winning(self)->None:
        """
        increase the number of winnings
        """
        self._winning_games+=1
       
    def get_legal_moves(self, board):
        """
        Return the legal moves
        """
        legal_moves = []

        rows, cols = board.shape
        # Top border indices
        top_indices = [(0, i) for i in range(cols)]

        # Bottom border indices
        bottom_indices = [(rows - 1, i) for i in range(cols)]

        # Left border indices (excluding corners)
        left_indices = [(i, 0) for i in range(1, rows - 1)]

        # Right border indices (excluding corners)
        right_indices = [(i, cols - 1) for i in range(1, rows - 1)]

        indices=top_indices+bottom_indices+left_indices+right_indices

        for x,y in indices:
            for direction in Move:
                if self.is_move_playable(board, (y,x), direction):
                                #print("back da playable")
                                legal_moves.append(((x, y), direction))
        #game.print()
        #print(legal_moves)
        return legal_moves

    def is_move_playable(self, board, position, direction):
        """
        check wheter the proposed move is applicable
        """
        x, y = position

        acceptable: bool = (
            # check if it is in the first row
            (x == 0 and y < 5)
            # check if it is in the last row
            or (x == 4 and y< 5)
            # check if it is in the first column
            or (x <5 and y ==0)
            # check if it is in the last column
            or (x <5 and y == 4)
            # and check if the piece can be moved by the current player
        ) 
        if acceptable is False:
            return False
        # Check if the move is within the bounds of the board
        if not (0 <= x < board.shape[0] and 0 <= y < board.shape[1]):
            return False

        if board[x, y] == 1-self.symbol:
            return False
        # Check if the move is towards an empty cell
        if direction == Move.TOP and x==0:
            #print("STEP 1")
            return False
        elif direction == Move.BOTTOM and x==4:
            #print("STEP 2")
            return False
        elif direction == Move.LEFT and y == 0:
            #print("STEP 3")
            return False
        elif direction == Move.RIGHT and y == 4:
            #print("STEP 4")
            return False

        return True

    def print_q_table(self):

        """
        print the q table
        """

        print("Printing first 5 rows...")
        for chiave, valore in list(self.q_table.items())[:10]:
            print(f'{chiave}: {valore} \n')

```

The Q table is updated following this formula:

![Alt text](image-6.png)

```python
 def update_q_table(self, trajectory, reward):
        """
        update the values in the q table
        """
        counter=0
        for state,action in trajectory:
            counter+=1

            new_state,_,_,rotation_type=self.transform_state(state)
            new_action=self.rotate_action(rotation_type, action)
            
            if (new_state) in self.q_table and new_action in self.q_table[new_state]:
               self.q_table[new_state][new_action]+=0.001* (sum(self.rewards) -  self.q_table[new_state][new_action])
                
```


```python
 def train(self,opponent):
        self._winning_games=0
        self.changing_symbol=False
        self.is_train=True
        num_iterations =100_000
        
        print("Montecarlo is training...")
        for i in tqdm(range(num_iterations)):

            # if i>(num_iterations/3)*2 and self.exploration_rate>0.1:
            #     self.exploration_rate=((1-self.exploration_rate)*10)/i
                #self.exploration_rate*=0.99
            game=MontecarloGame()
            
            if self.symbol==0:
                _,winner=game.play(self, opponent)
            else:
                _,winner=game.play(opponent, self)
    
            if i>num_iterations/2:
                self.changing_symbol=True
                self.symbol=1-self.symbol
                opponent.symbol=1-self.symbol
                
            if winner==self.symbol:
                self.add_winning()
        
        print("My player won ", self._winning_games/num_iterations)
        print(self.exploration_rate)

    def test(self, opponent):
        
        print("Montecarlo is testing...")
        self._winning_games=0
        self.is_train=False
        self.changing_symbol=False
        #self.symbol=0
        num_iterations=100

        self.exploration_rate=0
        for i in tqdm(range(num_iterations)):
            game=MontecarloGame()
            
            if self.symbol==0:
                _,winner=game.play(self, opponent)
            else:
                _,winner=game.play(opponent, self)
                
            if i>num_iterations/2:
                self.changing_symbol=True
                self.symbol=1-self.symbol
                opponent.symbol=1-self.symbol

            if winner==self.symbol:
                self.add_winning()
        
        print("My player won ", self._winning_games/num_iterations)
```

We defined a subclass of the game in order to implement the play method able to memorize the trajectory and update the values in the qtable.

```python
class MontecarloGame(Game):
    """
    a subclass of the game class that changes the play method to adapt to the montecarlo agent
    """
    def __move(self, from_pos, slide, index):
        return super()._Game__move(from_pos, slide, index)

    def play(self, player1: Player, player2: Player) -> int:

        trajectory=list()
        state=(set(), set())
        
        players=[player1,player2]
        #print(players)
        index=0
        while True:

            ok = False
            current_player=players[index]
            for x in range(super().get_board().shape[0]): 
                for y in range(super().get_board().shape[1]):
                    if super().get_board()[x][y]==0:
                        state[0].add((x,y))
                    elif super().get_board()[x][y]==1:
                        state[1].add((x,y))
            while not ok:
                from_pos, slide = current_player.make_move(self,state)
                #super().print()

                #print(from_pos,slide)
                ok = self.__move(from_pos, slide, current_player.symbol)
            
            move=(from_pos,slide)
            #print("player ", index, move)


            if(super().check_winner()!=-1):
                trajectory.append((deepcopy(state),move))
                break

            index=1-index

            trajectory.append((deepcopy(state),move))
            state=(set(), set())            

        if isinstance(player1, MontecarloAgent):
            final_reward, winner= (5, 0) if super().check_winner()==player1.symbol else (-5,1)
            #print(trajectory[-1])
            player1.rewards.append(final_reward)
            if player1.is_train:
                player1.update_q_table(trajectory,sum(player1.rewards))
                #player1.print_q_table()
            player1.rewards=[]

        elif isinstance(player2, MontecarloAgent):
            final_reward, winner= (5, 1) if super().check_winner()==player2.symbol else (-5,0)
            #print(trajectory[-1])
            player2.rewards.append(final_reward)
            
            if player2.is_train:
                player2.update_q_table(trajectory,sum(player2.rewards))
            player2.rewards=[]


        #print(self.get_board())
        return trajectory, winner

    def set_board(self,board):

        self._board=board

```

## QUIXO MAIN

```python
class RandomPlayer(Player):
    """class defining a player that chooses his moves randomly"""
    def __init__(self,symbol:None) -> None:
        super().__init__()
        self.symbol=symbol

    def make_move(self, game=None, state=None) -> tuple[tuple[int, int], Move]:

        #random.seed(time.time())
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move



class MyGame(Game):

    def set_board(self,board):

        self._board=board

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        for riga in self.get_board():
            for elemento in riga:
                if elemento == 0:
                    print("❌", end=" ")  # Simbolo per 0
                elif elemento == 1:
                    print("⭕️", end=" ")  # Simbolo per 1
                elif elemento == -1:
                    print("➖", end=" ")  # Simbolo per -1
            print("\n")

def train_montecarlo(agent: MontecarloAgent):

    """  with open('minmax_trained.pkl', 'rb') as file:
        agent.q_table=pickle.load(file) """
    agent.train(RandomPlayer(1-agent.symbol))
    # Save dictionary to a file
    with open('my_dict.pkl', 'wb') as file:
        pickle.dump(agent.q_table, file) 

def test_montecarlo(agent: MontecarloAgent):
    
    with open('my_dict.pkl', 'rb') as file:
        agent.q_table=pickle.load(file)
    #agent.print_q_table()

    opponent=RandomPlayer(1-agent.symbol)
    
    #opponent= MontecarloAgent(1-agent.symbol)
    #opponent.q_table=agent.q_table
    agent.test(opponent)


def minmax_simulation():
    
    ITERATIONS=50
    count=0


    SYMBOL_MYAGENT=0
    SYMBOL_OPPONENT=1-SYMBOL_MYAGENT
    DEPTH_MINMAX=2
    
 
    print("MINMAX DEPTH ", DEPTH_MINMAX)
    for i in tqdm(range(ITERATIONS)):

        player1 = MinmaxPlayer(DEPTH_MINMAX,SYMBOL_MYAGENT)
        player2 = RandomPlayer(SYMBOL_OPPONENT) 
        #player2 = MinmaxPlayer(3,SYMBOL_OPPONENT)
        g = MyGame()        
      
        winner = g.play( player1, player2)
        
        if winner==SYMBOL_MYAGENT:
            count+=1

    print("My player won ", count/ITERATIONS) 
    

    print("MINMAX DEPTH 4")
    for i in tqdm(range(ITERATIONS)):

        player1 = MinmaxPlayer(4,SYMBOL_MYAGENT)
        player2 = RandomPlayer(SYMBOL_OPPONENT) 

        g = Game()        
      
        winner = g.play(player1, player2)
        
        if winner==SYMBOL_MYAGENT:
            count+=1

    print("My player won ", count/ITERATIONS)
    
    count=0
    print("MINMAX DEPTH 5")
        
    for i in tqdm(range(ITERATIONS)):

        player1 = MinmaxPlayer(5,SYMBOL_MYAGENT)
        player2 = RandomPlayer(SYMBOL_OPPONENT) 

        g = Game()        
      
        winner = g.play(player1, player2)
        
        if winner==SYMBOL_MYAGENT:
            count+=1

    print("My player won ", count/ITERATIONS)
   


if __name__ == '__main__':

    #minmax_simulation()
    agent=MontecarloAgent(1)
    #agent2=MontecarloAgent(0)
    train_montecarlo(agent)
    agent=MontecarloAgent(0)
    #agent.print_q_table()
    test_montecarlo(agent)
