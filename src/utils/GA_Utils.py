import src.utils.MOEAD_Utils as MOEAD_Utils
import src.utils.Draw_Utils as Draw_Utils
import numpy as np
from joblib import Parallel, delayed

'''
遗传算法工具包
'''

def Creat_child(moead):
    # 创建一个个体
    child = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand(
        moead.Test_fun.Dimention)
    return child

def Creat_Pop(moead):
    # 创建moead.Pop_size个种群
    Pop = []
    Pop_FV = []
    if moead.Pop_size < 1:
        print('error in creat_Pop')
        return -1
    while len(Pop) != moead.Pop_size:
        X = Creat_child(moead)
        Pop.append(X)
        Pop_FV.append(moead.Test_fun.Func(X))
    moead.Pop, moead.Pop_FV = Pop, Pop_FV
    return Pop, Pop_FV

def mutate(moead, p1, p2, p3, F=0.5):
    # 改进为差分进化的变异策略
    var_num = moead.Test_fun.Dimention
    return p1 + F * (p2 - p3)

def crossover(moead, parent1, parent2, eta=30):
    # 改进为模拟二进制交叉（SBX）
    child1, child2 = np.copy(parent1), np.copy(parent2)
    for i in range(len(parent1)):
        if np.random.rand() <= 0.5:
            u = np.random.rand()
            beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
    return child1, child2

def select_by_crowding_distance(moead, population, fitness_values):
    # 计算拥挤距离并选择个体
    distances = np.zeros(len(population))
    sorted_indices = np.argsort(fitness_values, axis=0)
    max_fv = fitness_values[sorted_indices[-1]]
    min_fv = fitness_values[sorted_indices[0]]
    
    distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
    for i in range(1, len(population) - 1):
        distances[sorted_indices[i]] += (fitness_values[sorted_indices[i + 1]] - fitness_values[sorted_indices[i - 1]]) / (max_fv - min_fv)
    
    selected_indices = np.argsort(distances)[::-1][:len(population)]
    return [population[i] for i in selected_indices], [fitness_values[i] for i in selected_indices]

def elitism(moead, population, fitness_values, elite_size=5):
    # 精英保留策略，保留最优的个体
    elite_indices = np.argsort(fitness_values)[:elite_size]
    elites = [population[i] for i in elite_indices]
    elite_fv = [fitness_values[i] for i in elite_indices]
    return elites, elite_fv

def generate_next(moead, gen, wi, p0, p1, p2):
    # 使用差分变异和SBX交叉生成下一代
    qbxf_p0 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p0)
    qbxf_p1 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p1)
    qbxf_p2 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
    best = np.argmin(qbxf)
    
    Y1 = [p0, p1, p2][best]
    
    # 交叉与变异
    n_p0, n_p1 = crossover(moead, p0, p1)
    n_p1, n_p2 = crossover(moead, p1, p2)
    n_p0 = mutate(moead, p0, p1, p2)
    
    # 精英保留
    elites, elites_fv = elitism(moead, [n_p0, n_p1, n_p2], [qbxf_p0, qbxf_p1, qbxf_p2])
    n_p0, n_p1, n_p2 = elites

    qbxf_np0 = MOEAD_Utils.cpt_tchbycheff(moead, wi, n_p0)
    qbxf_np1 = MOEAD_Utils.cpt_tchbycheff(moead, wi, n_p1)
    qbxf_np2 = MOEAD_Utils.cpt_tchbycheff(moead, wi, n_p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2, qbxf_np0, qbxf_np1, qbxf_np2])
    best = np.argmin(qbxf)
    
    Y2 = [p0, p1, p2, n_p0, n_p1, n_p2][best]

    # 随机选择目标进行判断
    fm = np.random.randint(0, moead.Test_fun.Func_num)
    
    if moead.problem_type == 0 and np.random.rand() < 0.5:
        FY1 = moead.Test_fun.Func(Y1)
        FY2 = moead.Test_fun.Func(Y2)
        if FY2[fm] < FY1[fm]:
            return Y2
        else:
            return Y1
    return Y2

def envolution(moead):
    # 进化过程并行化
    for gen in range(moead.max_gen):
        moead.gen = gen
        
        def process_individual(pi, p):
            Bi = moead.W_Bi_T[pi]
            k = np.random.randint(moead.T_size)
            l = np.random.randint(moead.T_size)
            ik = Bi[k]
            il = Bi[l]
            Xi = moead.Pop[pi]
            Xk = moead.Pop[ik]
            Xl = moead.Pop[il]
            Y = generate_next(moead, gen, pi, Xi, Xk, Xl)
            cbxf_i = MOEAD_Utils.cpt_tchbycheff(moead, pi, Xi)
            cbxf_y = MOEAD_Utils.cpt_tchbycheff(moead, pi, Y)
            d = 0.001
            if cbxf_y < cbxf_i:
                moead.now_y = pi
                F_Y = moead.Test_fun.Func(Y)[:]
                MOEAD_Utils.update_EP_By_ID(moead, pi, F_Y)
                MOEAD_Utils.update_Z(moead, Y)
                if abs(cbxf_y - cbxf_i) > d:
                    MOEAD_Utils.update_EP_By_Y(moead, pi)
            MOEAD_Utils.update_BTX(moead, Bi, Y)
        
        Parallel(n_jobs=-1)(delayed(process_individual)(pi, p) for pi, p in enumerate(moead.Pop))

        if moead.need_dynamic:
            Draw_Utils.plt.cla()
            if moead.draw_w:
                Draw_Utils.draw_W(moead)
            Draw_Utils.draw_MOEAD_Pareto(moead, moead.name + ",gen:" + str(gen) + "")
            Draw_Utils.plt.pause(0.001)
        print(f'迭代 {gen}, 支配前沿个体数量 len(moead.EP_X_ID): {len(moead.EP_X_ID)}, moead.Z: {moead.Z}')
    return moead.EP_X_ID
