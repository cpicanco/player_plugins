# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import matplotlib.pyplot as plt
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

rx, ry = 10.5, 12.5

samples = 500
figure = plt.figure()
axes = figure.add_axes([.09, .09, .85, .85], axisbg=(.2, .2, .2, .3))
#axes = plt.subplots(111, figsize=(4,4)) 
figure.subplots_adjust(wspace=0.1,left=0.05, right=.98,bottom=0.1,top=0.92)
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['right'].set_visible(False)

# high accuracy, low precision
mu, sigma = 10, 2
x = np.random.normal(mu, sigma, samples)
y = np.random.normal(mu, sigma, samples)
axes.plot([mu,rx],[mu,ry], 'w--',label='Desvios em relação à P')
axes.scatter(x,y, marker='x',color='k',label='Baixa precisão')
axes.scatter(mu, mu, marker='o',color='w',s=25)
axes.annotate("Alta acurácia",
            xy=(mu,mu), xycoords='data',
            xytext=(22.5, 9), textcoords='data',
            size=12, va="center", ha="center",
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2",
                            color="w"), 
            )

# high precision, low accuracy
mu, sigma = 20, 0.5
x = np.random.normal(mu, sigma, samples)
y = np.random.normal(mu, sigma, samples)
axes.plot([mu,rx],[mu,ry], 'w--')
axes.scatter(x,y, marker='.',color='k',label='Alta precisão')
axes.scatter(mu, mu, marker='o',color='w',s=25)
axes.annotate("Baixa acurácia",
            xy=(mu,mu), xycoords='data',
            xytext=(22.5, 12), textcoords='data',
            size=12, va="center", ha="center",
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.2",
                            color="w"), 
            )


# real point
axes.scatter(rx, ry, marker='o',color='w',s=25)
axes.annotate("Ponto Real (P)",
            xy=(rx,ry), xycoords='data',
            xytext=(3, 15), textcoords='data',
            size=12, va="center", ha="center",
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.2",
                            color="w"), 
            )

legend = axes.legend(loc='upper left', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.70')

axes.set_xlabel('x (graus)')
axes.set_ylabel('y (graus)')

plt.ylim(ymin = 0)
plt.xlim(xmax=25, xmin=0)
plt.show()