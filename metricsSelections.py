from tetrisEnv import TetrisEnv
from agent import TetrisAgent
from helper import plot
import random
import json
lrs = [0.0005, 0.0001, 0.00005, 0.000025, 0.00001, 0.000005, 0.0000025, 0.000001, 0.0000005,
       0.00000025, 0.000000001]
def fitness(params : dict):
    agent = TetrisAgent(gamma=params['gamma'], learning_rate=params['lr'])
    metrics = {
        'drop' : params['drop'],
        'height' : params['height'],
        'bumpiness' : params['bumpiness'],
        'total_height' : params['total_height'],
        'holes' : params['holes']
    }
    env = TetrisEnv(metrics=metrics,render_mode="human")
    num_episodes = 1200
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    total_lines = 0
    for episode in range(1,num_episodes):
        
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)   
            next_state, reward, done, _, info = env.step(action)
            score = info['score']
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                total_lines += env.game.cleared_lines
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / episode
                plot_mean_scores.append(mean_score)
                if episode % 5 == 0:
                    try:
                        title = "gamma_" + str(agent.gamma) + "LR_" + str(agent.learning_rate)    
                        plot(plot_scores,plot_mean_scores,title)
                    except: pass
                break
            if score > record:
                record = score
                agent.save("models/tetris_dqn3_IOpiecesgamma"+str(agent.gamma)+".h5")
        agent.replay()
        print('Game', episode, "\nScore", score, '\nRecord: ', record, '\nEpsion: ', agent.epsilon, '\nMetrics', params)
        if episode % agent.update_target == 0:
            agent.update_target_model()
    return total_lines

class Cromosom:
    def __init__(self) -> None:
        self.gens = {
            'gamma': None,
            'lr': None,
            'drop' : None,
            'height' : None,
            'bumpiness' : None,
            'total_height' : None,
            'holes' : None
        }
        self.fitness_val = -1
        
    def randommize(self):
        self.gens['gamma'] = random.uniform(0.9, 0.999)
        self.gens['lr'] = random.sample(lrs,1)[0]
        self.gens['drop'] = random.randint(0, 6)
        self.gens['height'] = random.randint(1, 20)
        self.gens['bumpiness'] = random.randint(1, 20)
        self.gens['total_height'] = random.randint(1, 20)
        self.gens['holes'] = random.randint(1, 20)
    
    def calc_fitness(self):
        self.fitness_val = fitness(self.gens)
        return self.fitness_val
        
    def mutate(self):
        match random.randint(0,6):
            case 0:
                self.gens['gamma'] = random.uniform(0.9, 0.999)
            case 1:
                self.gens['lr'] = random.sample(lrs, 1)[0]
            case 2:
                self.gens['drop'] = random.randint(0, 6)
            case 3:
                self.gens['height'] = random.randint(1, 20)
            case 4:
                self.gens['bumpiness'] = random.randint(1, 20)
            case 5:
                self.gens['total_height'] = random.randint(1, 20)
            case 6:
                self.gens['holes'] = random.randint(1, 20)
                
    def getChildGens(self, gens2: dict):
        return {
            'gamma': self.gens['gamma'] if random.randint(0,1)==1 else gens2['gamma'],
            'lr': self.gens['lr'] if random.randint(0,1)==1 else gens2['lr'],
            'drop' : self.gens['drop'] if random.randint(0,1)==1 else gens2['drop'],
            'height' : self.gens['height'] if random.randint(0,1)==1 else gens2['height'],
            'bumpiness' : self.gens['bumpiness'] if random.randint(0,1)==1 else gens2['bumpiness'],
            'total_height' : self.gens['total_height'] if random.randint(0,1)==1 else gens2['total_height'],
            'holes' : self.gens['holes'] if random.randint(0,1)==1 else gens2['holes']
        }
        
class GeneticAlg:
    def __init__(self) -> None:
        self.population : list[Cromosom] = []
        self.pop_lenght = 10
        for i in range(self.pop_lenght):
            crom = Cromosom()
            crom.randommize()
            self.population.append(crom)
    
    def run(self, epocs):
        final_selection = {}
        for _ in range(epocs):
            # calc fitnesses
            for crom in self.population:
                crom.calc_fitness()
            self.population.sort(key=lambda x: x.fitness_val, reverse=True)
            final_selection = {
                "selections":[
                    self.population[0].gens,
                    self.population[1].gens,
                    self.population[2].gens,
                    self.population[3].gens
                ]
            }
            #selection
            new_population : list[Cromosom] = []
            new_population.extend(self.population[:4])
            
            #get childs
            childs : list[Cromosom] = []
            for i in range(len(new_population)-1):
                for j in range(i+1, len(new_population)):
                    new_crom_gens = new_population[i].getChildGens(new_population[j].gens)
                    new_crom = Cromosom()
                    new_crom.gens = new_crom_gens
                    childs.append(new_crom)
            new_population.extend(childs)
            self.population = new_population
            
            #mutate
            for crom in self.population:
                if random.randint(0,1) == 0:
                    crom.mutate()
        return final_selection
                    
def main():
    gen_alg = GeneticAlg()
    selection = gen_alg.run(8)
    print(selection)
    json.dump(selection, open( "metrics.json", 'w' ))
                    
if __name__ == "__main__":
    main()
