import numpy as np
import matplotlib.pyplot as plt
import math

plt.figure(figsize=(12, 6))

x20 = np.array([253.83 , 234.57 , 212.80 , 192.02 , 180.15 , 164.54 , 149.23 , 135.26 , 122.63 , 109.57])
y20 = np.array([512 , 476 , 432 , 388 , 368 , 332 , 304 , 272 , 248 , 220 ])
plt.errorbar(x20,y20,xerr=0.002*x20+0.02,yerr=1.2,linestyle='',ecolor='red',linewidth=6)
p, v = np.polyfit(x20, y20, deg=1, cov=True)
x=np.arange(min(x20),max(x20),0.01)
plt.plot(x,x*p[0]+p[1],color='red',label='l = 20 см',linewidth=1)

x = 0
y = 0
xy=0
for i in range(len(x20)):
    x+=float(x20[i])**2
    xy+=float(x20[i])*float(y20[i])
    y+=float(y20[i])**2
print(((y/x-xy**2/x**2)/10)**0.5)
print(p[0]*((1.2/max(y20))**2+((0.002*max(x20)+0.02)/max(x20))**2)**0.5)
print(((((y/x-xy**2/x**2)/10)**0.5)**2+(p[0]*((1.2/max(y20))**2+((0.002*max(x20)+0.02)/max(x20))**2)**0.5)**2)**0.5)
print(p[0]*(1+p[0]/4000))
print()


x30=np.array([186.24 , 172.77 , 162.92 , 152.82 , 142.27 , 130.37 , 121.4 , 111.70 , 101.19 , 88.60])
y30=np.array([568 , 528 , 500 , 468 , 436 , 396 , 368 , 340 , 308 , 272])
plt.errorbar(x30,y30,xerr=0.002*x30+0.02,yerr=1.2,linestyle='',ecolor='green',linewidth=6)
p, v = np.polyfit(x30, y30, deg=1, cov=True)
x=np.arange(min(x30),max(x30),0.01)
plt.plot(x,x*p[0]+p[1],color='green',label='l = 30 см',linewidth=1)

x = 0
y = 0
xy=0
for i in range(len(x30)):
    x+=float(x30[i])**2
    xy+=float(x30[i])*float(y30[i])
    y+=float(y30[i])**2
print(((y/x-xy**2/x**2)/10)**0.5)
print(p[0]*((1.2/max(y30))**2+((0.002*max(x30)+0.02)/max(x30))**2)**0.5)
print(((((y/x-xy**2/x**2)/10)**0.5)**2+(p[0]*((1.2/max(y30))**2+((0.002*max(x30)+0.02)/max(x30))**2)**0.5)**2)**0.5)
print(p[0]*(1+p[0]/4000))
print()

x50=np.array([113.6 , 108.97 , 102.84 , 96.28 , 89.7 , 84.73 , 78.69 , 71.64 , 66.83 , 61.05])
y50=np.array([ 588 , 560 , 532 , 496 , 464 , 436 , 404 , 368 , 344 , 312])
plt.errorbar(x50,y50,xerr=0.002*x50+0.02,yerr=1.2,linestyle='',ecolor='blue',linewidth=6)
p, v = np.polyfit(x50, y50, deg=1, cov=True)
x=np.arange(min(x50),max(x50),0.01)
plt.plot(x,x*p[0]+p[1],color='blue',label='l = 50 см',linewidth=1)

x = 0
y = 0
xy=0
for i in range(len(x50)):
    x+=float(x50[i])**2
    xy+=float(x50[i])*float(y50[i])
    y+=float(y50[i])**2
print(((y/x-xy**2/x**2)/10)**0.5)
print(p[0]*((1.2/max(y50))**2+((0.002*max(x50)+0.02)/max(x50))**2)**0.5)
print(((((y/x-xy**2/x**2)/10)**0.5)**2+(p[0]*((1.2/max(y50))**2+((0.002*max(x50)+0.02)/max(x50))**2)**0.5)**2)**0.5)
print(p[0]*(1+p[0]/4000))

plt.xlabel('I, мА')
plt.ylabel('U, мВ')
plt.legend(loc='best', fontsize=12)
plt.savefig('графики.png')

plt.show()
