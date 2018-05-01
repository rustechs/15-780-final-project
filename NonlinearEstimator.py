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
		u += [t.strip('\n').split()[0:3]]
		x += [t.strip('\n').split()[3:9]]
		xdot += [t.strip('\n').split()[9:15]]
		y += [t.strip('\n').split()[15:]]
	return u,x,xdot,y,int(deg)

def main(ifile):
	u,x,xdot,y,deg = Input(ifile)
	print('Last line: u:{} x:{} xdot:{} y:{}'.format(u[-1],x[-1],xdot[-1],y[-1]))
	estimator = SI(degree=deg, data=[u,x,xdot,y], form=[3,6,6,3])
	time, model = estimator.solver('ml_regression')
	[tempA,tempB,tempC,tempD,TempE] = model
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	main(ifile)