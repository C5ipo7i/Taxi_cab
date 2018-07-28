import numpy as np

def dayofweek(d,m,y):
    t = [0,3,2,5,0,3,
    5,1,4,6,2,4]
    y -= m < 3
    #step1 = y + int(y / 4)
    #step2 = step1 - int(y / 100)
    #step3 = step2 + int(y/400)
    #step4 = step3 + t[m-1]
    #step5 = step4 + d
    #step6 = step5 % 7
    return ((y + int(y / 4) - int(y / 100) + int(y/400) + t[m-1] + d) % 7)

def vectorized_dayofweek(d,m,y):
    #t = np.tile(np.array([0,3,2,5,0,3,5,1,4,6,2,4]),(d.shape[0],1))
    t = np.array([0,3,2,5,0,3,5,1,4,6,2,4])
    y = np.subtract(y,m<3)
    #step1 = np.divide(y,4).astype(int)
    #step2 = np.divide(y,100).astype(int)
    #step3 = np.divide(y,400).astype(int)
    #step4 = np.add(np.subtract(np.add(y,step1),step2),step3)
    #step5 = np.add(step4,t[m-1])
    #step6 = np.add(step5,d)
    #step7 = np.mod(step6,7)
    return np.mod(np.add(np.add(np.add(np.subtract(np.add(y,np.divide(y,4).astype(int)),np.divide(y,100).astype(int)),np.divide(y,400).astype(int)),t[m-1]),d),7)


#years = np.array([2009,2010,2011,2012])
#months = np.array([6,1,8,4])
#days = np.array([15,5,18,21])

#new_days = vectorized_dayofweek(days,months,years)
#print(new_days)
#day = dayofweek(21,4,2012)
#print(day)