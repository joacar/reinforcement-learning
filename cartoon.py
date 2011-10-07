import numpy
import pylab

test = (0, 1, 3, 2, 5, 2)

images = (pylab.imread('step1.png'),
          pylab.imread('step2.png'),
          pylab.imread('step3.png'),
          pylab.imread('step4.png'),
          pylab.imread('step5.png'),
          pylab.imread('step6.png'),
          pylab.imread('step7.png'),
          pylab.imread('step8.png'),
          pylab.imread('step9.png'),
          pylab.imread('step10.png'),
          pylab.imread('step11.png'),
          pylab.imread('step12.png'),
          pylab.imread('step13.png'),
          pylab.imread('step14.png'),
          pylab.imread('step15.png'),
          pylab.imread('step16.png'))

comic = numpy.concatenate([images[i] for i in test], axis=1)

pylab.imshow(comic)
pylab.show()
