
import numpy as np
import torch

DNA_SIZE = 16
POP_SIZE = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 10

class GA(object):

    def setF(self,func):
        self.F = func


    def get_fitness(self,pop): 
        weight = self.translateDNA(pop)
        pred = []
        for i in range(len(weight)):
            pred.append(self.F(weight[i]))
        return (pred - np.min(pred)) + 1e-3 #减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
    
    
    def translateDNA(self,pop): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    	weights = []
        for i in range(np.size(pop,0)):
            weight =  torch.tensor(pop[i],dtype= torch.double)
            weight.view(2,-1)
            weights.append(weight)
    	return weights
    
    def crossover_and_mutation(self,pop, CROSSOVER_RATE = 0.8):
    	new_pop = []
    	for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
    		child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
    		if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
    			mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
    			cross_points = np.random.randint(low=0, high=DNA_SIZE*2)	#随机产生交叉的点
    			child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
    		self.mutation(child)	#每个后代有一定的机率发生变异
    		new_pop.append(child)
    
    	return new_pop

    def mutation(self,child, MUTATION_RATE=0.003):
    	if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
    		mutate_point = np.random.randint(0, DNA_SIZE*2)	#随机产生一个实数，代表要变异基因的位置
    		child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转
    
    def select(self,pop, fitness):    # nature selection wrt pop's fitness
        idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                               p=(fitness)/(fitness.sum()) )
        return pop[idx]
    
    def print_info(self,pop):
    	fitness = self.get_fitness(pop)
    	max_fitness_index = np.argmax(fitness)
    	print("max_fitness:", fitness[max_fitness_index])
    	weights = self.translateDNA(pop)
    	print("best gene：", pop[max_fitness_index])
    	print("weights:", weights[max_fitness_index])
        
        return weights[max_fitness_index],fitness[max_fitness_index]
      

if __name__ == "__main__":
    ga = GA()
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*2)) #matrix (POP_SIZE, DNA_SIZE)
    pop.size
    for _ in range(N_GENERATIONS):#迭代N代
        weights = ga.translateDNA(pop)
        pop = np.array(ga.crossover_and_mutation(pop, CROSSOVER_RATE))
        #F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = ga.get_fitness(pop)
        pop = ga.select(pop, fitness) #选择生成新的种群
	
	good_weight ,best_acc = ga.print_info(pop)

