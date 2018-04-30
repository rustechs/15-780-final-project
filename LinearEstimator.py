from SystemIdentification import *
import sys

def Input(fname):
	fhandle = open(fname, 'r')
	temp = fhandle.readlines()[2:]
	u=[]
	x=[]
	xdot=[]
	y=[]
	for t in temp:
		u += [t.strip('\n').split()[0:3]]
		x += [t.strip('\n').split()[3:9]]
		xdot += [t.strip('\n').split()[9:15]]
		y += [t.strip('\n').split()[15:]]
	return u,x,xdot,y


def main(ifile, ofile):
	u,x,xdot,y = Input(ifile)
	print('Last line: u:{} x:{} xdot:{} y:{}'.format(u[-1],x[-1],xdot[-1],y[-1]))
	estimator = SI(degree=1, data=[u,x,xdot,y], form=[3,6,6,3])
	time, model = estimator.solver('ml_regression')
	[tempA,tempB,tempC,tempD] = model
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	ofile = sys.argv[2]
	main(ifile, ofile)