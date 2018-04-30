from SystemIdentification import *
import sys

def Input(fname):
	fhandle = open(fname, 'r')
	temp = fhandle.readlines()[1:]
	u=[]
	x=[]
	xdot=[]
	y=[]
	for t in temp:
		u += [t.strip('\n').split()[0:3]]
		x += [t.strip('\n').split()[3:6]]
		xdot += [t.strip('\n').split()[6:9]]
		y += [t.strip('\n').split()[9:]]
	return x,y

def main(ifile, ofile):
	x,y = Input(ifile)
	print('First line: u:{} x:{} xdot:{} y:{}'.format(x[0],y[0]))
	estimator = SI(degree=1, xdata=x, ydata=y)
	estimator.augmentInput()
	estimator.solver(type)
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	ofile = sys.argv[2]
	main(ifile, ofile)