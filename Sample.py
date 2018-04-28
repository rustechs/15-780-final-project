import SystemIdentification


def main(ifile, ofile):
	estimator = SI(input_data_file=ifile)
	estimator.augmentInput()
	estimator.solver(type)
	estimator.getOutput()

if __name__=='__main__':
	ifile = sys.argv[1]
	ofile = sys.argv[2]
	main(ifile, ofile)