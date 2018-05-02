from SystemIdentification import *
import sys

def Input(fname):
	fhandle = open(fname, 'r')
	lines=fhandle.readlines()
	deg = lines[0].strip('\n').strip('\r').split()[-1]
	print('deg is {}'.format(deg))
	temp = lines[2:]
	x=[]
	xdot=[]
	y=[]
	ytrue=[]
	for t in temp:
		x += [t.strip('\n').split()[3:5]]
		xdot += [t.strip('\n').split()[5:7]]
		y += [t.strip('\n').split()[7]]
		ytrue += [t.strip('\n').split()[9]]
	return x,xdot,y,int(deg),ytrue

def main(ifile):
	x,xdot,y,deg,ytrue = Input(ifile)
	print('Last line: u: x:{} xdot:{} y:{} ytrue:{}'.format(x[-1],xdot[-1],y[-1], ytrue[-1]))
	estimator = SI(degree=deg, data=[x,xdot,y], form=[2,2,2], ytrue=ytrue)
	time, model = estimator.solver('robust')
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	main(ifile)