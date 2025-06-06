from ast import Not
from copy import deepcopy
import genericpath
import numpy as np
import random

class Individual:
    def __init__(self, gene, comp):
        self.gene, self.gender, self.competitiveness = self.birth(gene, comp)
        if self.gender == 'm':
            self.cpTime = int(self.survival_time() * 5)
        else:
            self.cpTime = int(self.survival_time() * 0.8)
        self.childNum = 0

    @classmethod
    def birth(cls, gene, comp):
        g = []
        for i in gene:
            if len(i) > 2:
                if isinstance(i[0].gender, str):
                    g1 = random.choice(i)
                    while g1.gender == 'm':
                        g1 = random.choice(i)
                    g2 = random.choice(i)
                    g.append([g1, g2])
                else:
                    g.append(random.sample(i, 2))
            else:
                g.append(i)
        if 'm' in [g[-1][0].gender, g[-1][1].gender] and isinstance(g[-1][1].gender, str):
            gender = 'm'
        else:
            gender = 'f'

        comp = max(0, (min(comp + np.random.normal() / 50, 1)))
        return g, gender, comp

    @classmethod
    def coitus(cls, g1, g2, parentComp1, parentComp2):
        newGene = []
        for allele in zip(g1, g2):
            newGene.append([random.choice(allele[0]), random.choice(allele[1])])
        return newGene, 0.5 * (parentComp1 + parentComp2)

    def survival_time(self):
        survival = 0
        for gene in self.gene:
            if gene[0].dominant > gene[1].dominant:
                survival += gene[0].survival
                if gene[0].survival == 0:
                    survival = 0
                    break
            else:
                survival += gene[1].survival
                if gene[1].survival == 0:
                    survival = 0
                    break
        survival /= len(self.gene)

        if survival < 15:
            return 0
        else:
            return int(survival - 15)


class Environment:
    def __init__(self, pnormal=0.9, evolen=1000):
        self.pnormal = pnormal
        self.evolen = evolen
    
    def forward(self, population, ceil):
        pop = dict()
        for i in population:
            pop[i.competitiveness] = i
        pop = list(dict(sorted(pop.items(), key=lambda x : x[0], reverse=True)).values())

        while len(pop) > ceil:
            for pp in pop[::-1]:
                if np.random.random() > pp.competitiveness:
                    pop.remove(pp)
                    break

        cpList = self.cp(pop)
        newpop = []
        for cps in cpList:
            oosperm = Individual.coitus(cps[0].gene, cps[1].gene, cps[0].competitiveness, cps[1].competitiveness)
            newpop.append(Individual(*oosperm))
        return newpop

    def evolution(self, population, populationNumber, genderFreq):
        generations = [population]
        gid = 1
        for _ in range(self.evolen):
            for i in generations[-1]:
                i.competitiveness = max(0, (min(np.random.normal() / 20 + 0.5, 1)))
            ceil = max(200, np.random.normal() * 300 + 1000)
            generations.append(self.forward(generations[-1], ceil))
            populationNumber[gid] = len(generations[-1])
            for p in range(len(generations[-1])):
                if generations[-1][p].gender == 'm':
                    genderFreq['m'][gid] += 1
                else:
                    genderFreq['f'][gid] += 1
            print(ceil, gid, populationNumber[gid], genderFreq['m'][gid], genderFreq['f'][gid])
            gid += 1
        return generations
    
    def cp(self, sortedPopulation):
        affair = 0.7
        cpList = []
        for p1 in sortedPopulation:
            if p1 == sortedPopulation[-1]:
                break
            if p1.childNum >= p1.cpTime:
                continue
            for p2 in sortedPopulation:
                if p1.gender == p2.gender or p2.childNum >= p2.cpTime or np.random.random() > affair:
                    continue
                p1.childNum += 1
                p2.childNum += 1
                cpList.append((p1, p2))
        return cpList

    

class gene:
    def __init__(self, gender):
        self.survival = max(0, 30 * (min(np.random.normal() / 2 + 1, 100)))
        self.dominant = np.random.random()
        if gender:
            intGender = np.random.randint(0, 2)
            if intGender == 0:
                self.gender = 'm'
            else:
                self.gender = 'f'
        else:
            self.gender = None

def main():
    populationSize = 100
    genderBlindGeneNum = 100
    pnormal = 0.9
    evolen = 50

    genePool = []
    gidMap = dict()

    genderFreq = {'m' : np.zeros(evolen + 1), 'f' : np.zeros(evolen + 1)}
    populationNumber = np.zeros(evolen + 1)
    populationNumber[0] = populationSize

    id = 0
    for i in range(genderBlindGeneNum + 1):
        if i < genderBlindGeneNum:
            gender = False
        else:
            gender = True
        geneNum = max(3, np.random.randint(100))
        genePool.append([])
        for _ in range(geneNum):
            g = gene(gender)
            genePool[i].append(g)
            gidMap[id] = (g, id)
            id += 1
    geneFreq = np.zeros((len(gidMap), evolen + 1))

    population = []
    for _ in range(populationSize):
        population.append(Individual(genePool, 0.5))
        if population[-1].gender == 'm':
            genderFreq['m'][0] += 1
        else:
            genderFreq['f'][0] += 1
    env = Environment(pnormal, evolen)
    generations = env.evolution(population, populationNumber, genderFreq)

    for gener in range(len(generations)):
        for p in generations[gener]:
            for geGroup in p.gene:
                for ge in geGroup:
                    if ge.gender == 'f':
                    # if 1:
                        for gid in gidMap:
                            if ge in gidMap[gid]:
                                geneFreq[gid, gener] += 1
        geneFreqGener = dict(zip(range(len(geneFreq)), geneFreq[:, gener]))
        geneFreqGener = sorted(geneFreqGener.items(), key=lambda x : x[1], reverse=True)[:10]
        for i in range(5):
            print('id', geneFreqGener[i][0], 'freq', geneFreqGener[i][1], 
            'gender', gidMap[geneFreqGener[i][0]][0].gender, 'survival', gidMap[geneFreqGener[i][0]][0].survival, 
            'dominant', gidMap[geneFreqGener[i][0]][0].dominant)
        print('\n')
main()