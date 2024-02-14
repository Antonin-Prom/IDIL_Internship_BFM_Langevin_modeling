import numpy as np





def first_passage(frequency,traj,dt):
    
    """Compute the first passage time by iterating through an array of lag times

    Args:
        frequency (float): The targeted frequency.
        traj (array): The targeted array of positions.
        

    Returns:
       
    """
    max_minima_k = int(traj[len(traj)-1]/(np.pi/frequency))
    minimas = np.arange(0, max_minima_k, np.pi/frequency)
    passage_time = np.zeros(max_minima_k)
    counter = 0
    
    for n in range(len(traj)):
        m = int(traj[n] / (np.pi/frequency))
        while traj[n] / (np.pi/frequency) < m + 1:
            n += 1
        else:
            passage_time[counter] = n*dt
            counter += 1
            
    mean_pass_time = np.mean(passage_time[:-1] - passage_time[1:])
    return mean_pass_time


            
        