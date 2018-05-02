from SystemIdentification import *
import sys

def Input(fname):
	fhandle = open(fname, 'r')
	lines=fhandle.readlines()
	deg = lines[0].strip('\n').strip('\r').split()[-1]
	print('deg is {}'.format(deg))
	temp = lines[2:]
	u=[]
	x=[]
	xdot=[]
	y=[]
	for t in temp:
		x += [t.strip('\n').split()[0:6]]
		xdot += [t.strip('\n').split()[6:12]]
		y += [t.strip('\n').split()[12:]]
	return x,xdot,y,int(deg)

def main(ifile):
	x,xdot,y,deg = Input(ifile)
	print('Last line: u: x:{} xdot: y:{}'.format(x[-1],y[-1]))
	estimator = SI(degree=deg, data=[x,xdot,y], form=[6,6,3])
	time, model = estimator.solver('ml_regression')
	estimator.getOutput()
	time, model = estimator.solver('l1')
	estimator.getOutput()
	time, model = estimator.solver('l2')
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	main(ifile)