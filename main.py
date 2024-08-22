from mo_pg import MOPG
from mo_tsivr_pg import MOTSIVRPG
from cfg import Config
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import concurrent.futures
import os
import sys

def train(algorithm, id):
    total_rewards = []
    total_rewards.append(algorithm.evaluate())

    for epoch in tqdm(range(Config.epochs)):

        if Config.debug:
            print(total_rewards)

        algorithm.train_step()
        total_rewards.append(algorithm.evaluate())
        if Config.checkpoint and (epoch + 1) % Config.checkpoint_interval == 0:
            save_data(total_rewards, epoch + 1, id)

    return total_rewards

def create_and_train(id, M): 
    seed = (8+id) * 10**7

    Config.initalize_env(M)

    alg_name = Config.alg_name
    if alg_name == 'MOTSIVRPG':
        algorithm = MOTSIVRPG(seed=seed)
    elif alg_name == 'MOPG':
        algorithm = MOPG(seed=seed)

    return train(algorithm, id)

def main(M):

    Config.initalize_env(M)

    if Config.parallel:      
        with concurrent.futures.ProcessPoolExecutor() as executor: 
            total_rewards = [executor.submit(create_and_train, id, M) for id in range(Config.num_runs)]
            total_rewards = [el.result() for el in total_rewards]
    else:
        for id in range(Config.num_runs):
            total_rewards = create_and_train(id)

    save_data(total_rewards)
    
    Config.env.close()
    if Config.debug:
        print("Training completed")

    if Config.parallel:      
        total_rewards = np.median(total_rewards, axis=0)
        

def save_data(total_rewards, epoch=None, id=None):
    total_rewards = np.array(total_rewards)

    optional_info = ''

    if Config.env_name == 'queue': 
        optional_info += '_H=' + str(Config.H) + '_M=' + str(len(Config.env.arrival_rates))

    if epoch is not None:
        optional_info += '_CHECKPOINT-epoch=' + str(epoch)
    if id is not None:
        optional_info += '_thread-id=' + str(id)

    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    np.save('./results/' + Config.env_name + '_' + Config.alg_name + optional_info, total_rewards)

    if Config.debug and epoch is not None and id is not None:
        print("epoch: ----------------------------------------" + str(epoch))
        print("id: ----------------------------------------" + str(id))


if __name__ == "__main__":
    if len(sys.argv) == 0:
        main()
    else:
        for M in sys.argv:
            main(int(M))
