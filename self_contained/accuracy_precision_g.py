# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import matplotlib.pyplot as plt
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

samples = 1000
figure = plt.figure()
axes = figure.add_axes([0.1, 0.1, 0.8, 0.8], frameon = 0)

# high accuracy, low precision
mu, sigma = 10, 2
x = np.random.normal(mu, sigma, samples)
y = np.random.normal(mu, sigma, samples)
axes.scatter(x,y, marker='x',color='k',label='Alta acurácia, baixa precisão')

# high precision, low accuracy
mu, sigma = 20, 1
x = np.random.normal(mu, sigma, samples)
y = np.random.normal(mu, sigma, samples)
axes.scatter(x,y, marker='.',color='k',label='Baixa acurácia, alta precisão')

# real point
x, y = 10.5, 10.5
axes.plot([0,x,x],[y,y,0], 'k--', label='Ponto Real (Círculo Branco)')
axes.scatter(x, y, marker='o',color='w',s=100)
axes.legend(loc=(0.0,0.73))
axes.set_xlabel('x (graus)')
axes.set_ylabel('y (graus)')

plt.ylim(ymin = 0)
plt.xlim(xmin=0)
plt.show()