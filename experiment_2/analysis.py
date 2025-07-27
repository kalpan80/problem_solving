import matplotlib.pyplot as plt
import pandas as pd

def exploration():
    df = pd.DataFrame(columns=['Episode', 'RL_Exploration', 'LLM_Exploration'])
    rl = pd.read_csv('taxi_training_q.csv')
    llm = pd.read_csv('llm_training.csv')
    df['Episode'] = llm['Episode']
    df['RL_Exploration'] = rl['Exploration']
    df['LLM_Exploration'] = llm['Exploration']

    df.plot(x='Episode',y=['RL_Exploration','LLM_Exploration'])
    plt.title('Line Plot of RL_Exploration and LLM_Exploration')
    plt.xlabel('Episodes')
    plt.ylabel('RL_Exploration and LLM_Exploration')
    plt.show()

def exploitation():
    df = pd.DataFrame(columns=['Episode', 'RL_Exploitation', 'LLM_Exploitation'])
    rl = pd.read_csv('taxi_training_q.csv')
    llm = pd.read_csv('llm_training.csv')
    df['Episode'] = llm['Episode']
    df['RL_Exploitation'] = rl['Exploitation']
    df['LLM_Exploitation'] = llm['Exploitation']

    df.plot(x='Episode',y=['RL_Exploitation','LLM_Exploitation'])
    plt.title('Line Plot of RL_Exploitation and LLM_Exploitation')
    plt.xlabel('Episodes')
    plt.ylabel('RL_Exploitation and LLM_Exploitation')
    plt.show()

def steps():
    df = pd.DataFrame(columns=['Episode', 'RL_Steps', 'LLM_Steps'])
    rl = pd.read_csv('taxi_infer_q.csv')
    llm = pd.read_csv('llm_infer.csv')
    df['Episode'] = llm['Episode']
    rl.loc[rl['Steps'] == 200, 'Steps'] = 0
    llm.loc[llm['Steps'] == 200, 'Steps'] = 0

    df['RL_Steps'] = rl[rl['Steps'] < 200]['Steps']
    df['LLM_Steps'] = llm[llm['Steps'] < 200]['Steps']

    df.loc[df['RL_Steps'] > 0, 'RL_Completed'] = 1
    df.loc[df['LLM_Steps'] > 0, 'LLM_Completed'] = 1

    df.loc[df['RL_Completed'] != 1, 'RL_Completed'] = 0
    df.loc[df['LLM_Completed'] != 1, 'LLM_Completed'] = 0

    df['RL_Completed'] = df['RL_Completed'].cumsum()
    df['LLM_Completed'] = df['LLM_Completed'].cumsum()

    # df.to_csv('steps.csv')

    df.plot(x='Episode',y=['RL_Completed','LLM_Completed'])
    plt.title('Line Plot of Successful Completion of Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Successful Episodes')
    plt.show()

def epsilon():
    df = pd.DataFrame(columns=['Episode', 'RL_EPSILON', 'LLM_EPSILON'])
    rl = pd.read_csv('taxi_training_q.csv')
    llm = pd.read_csv('llm_training.csv')
    df['Episode'] = llm['Episode']
    df['RL_EPSILON'] = rl['epsilon']
    df['LLM_EPSILON'] = llm['epsilon']

    df.plot(x='Episode',y=['RL_EPSILON','LLM_EPSILON'])
    plt.title('Line Plot of RL_EPSILON and LLM_EPSILON')
    plt.xlabel('Episodes')
    plt.ylabel('RL_EPSILON and LLM_EPSILON')
    plt.show()


steps()
epsilon()
exploration()
exploitation()