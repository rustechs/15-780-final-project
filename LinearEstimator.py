from SystemIdentification import *
import sys

def Input(fname):
	fhandle = open(fname, 'r')
	temp = fhandle.readlines()[1:]
	x=[]
	y=[]
	for t in temp:
		x += [t.strip('\n').split()[:3]]
		y += [t.strip('\n').split()[3:6]]
	return x,y

def main(ifile, ofile):
	x,y = Input(ifile)
	print('First line: {} {}'.format(x[-1],y[-1]))
	estimator = SI(degree=1, xdata=x, ydata=y)
	model = estimator.solver('ml_regression')
	#print(model[1])
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	ofile = sys.argv[2]
	main(ifile, ofile)