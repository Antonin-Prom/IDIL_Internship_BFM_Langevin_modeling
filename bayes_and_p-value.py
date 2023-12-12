import matplotlib.pyplot as plt
import numpy as np

""" Bayse Theorem """
def bayse(PH,PDH,p_value):
    return(PH*PDH/(PH*PDH+(1-PH)*p_value))

"""
Set of possible initial credence in the hypothesis:
P(H) = 0.001 
would be a situation in wich you think this hypothesis is very unlikely 
for example if H = (Chloroquine heal immediatly every patients from covid, cancer, poverty)
P(H) = 0.5
This credence is coherent with the fact that chloroquine had good results in vitro
"""

H = np.linspace(0,1,50)


"""
If the data in a controled randomised trial is : 
a % of recovery of 35% in the control group
a % of recovery of 40% in the trial group
"""


"""
P(D|H)
Considering that we know that the hypothesis is true, are the results from our experiment likely to happen ?
P(D|H) = 0.01 

would signify that if the hypothesis is true it's almost impossible to obtain those data from the experiment.
So if P(D|H) is low it reduces the probability that our hypothesis given those data is a good hypothesis.

For example :
If our hypothesis is that chloroquine is effective to treat covid19,
it is not likely that we obtain as a result in our random controled trial: 
"Each patient to wich we gave chloroquine dies and every patient of the control group survives."
"""

"""
Given our data and our hypothesis that cholorquine is effective 
assuming that a treatment is effective if the trial group perform 10% better than the controled
P(D|H) = 0.5
"""

PDH = 0.5

"""
P(D)
Corresponds to the total probability of obtaining those data, condidering that the hypothesis is true or false
P(D) =  P(H)P(D|H) + P(NON H)P(D|NON H)
P(D|NON H) this term is corresponding to the p-value.
we can write 
p(D) = P(H)P(D|H) + P(H)*p-value
We can check that if the term corresponding to the probability of obtaining those data when H is false is great than p(H|D) is lower. same with a great p(D) in general.
It's easy to see that P(D) must be greater than P(D|H).
"""



plt.plot(H,bayse(H,PDH,0.01),label = ' PDH = 0.5, p_value = 0.01')
plt.plot(H,bayse(H,PDH,0.05),label = ' PDH = 0.5, p_value = 0.05')
plt.plot(H,bayse(H,PDH,0.2),label = ' PDH = 0.5, p_value = 0.2')
plt.title("post-trial probability of effective medecine function of pre-trial prior")
plt.legend(loc='lower right')
plt.ylabel("P(H|D)")
plt.xlabel("P(H)")
plt.show()